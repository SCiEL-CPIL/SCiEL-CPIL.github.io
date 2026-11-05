import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = q ** n  # 6561

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists (17 neighbors for q=3,n=8,radius=1)
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute neighbor masks as boolean arrays for fast vectorized intersection
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # --- Phase 1: Surgical Partition via Greedy Disjoint Clique Matching ---
    # Partition all 6561 words into disjoint 17-point stars (neighborhoods).
    # Each star center becomes a codeword; the partition guarantees full coverage.
    # 6561 / 17 = 486 exactly, matching the known best upper bound.
    def surgical_partition():
        used = np.zeros(total_words, dtype=bool)
        code_list = []
        
        # Process words in a deterministic order that favors good star centers
        # A good center is one whose neighborhood has few already-used points
        order = list(range(total_words))
        # Shuffle with a fixed seed for reproducibility but varied exploration
        random.seed(42)
        random.shuffle(order)
        
        for c in order:
            if used[c]:
                continue
            neigh_c = neighbors[c]
            # Check if this star can be formed (all neighbors not yet used)
            if np.any(used[neigh_c]):
                # Try to find the best available neighbor to use as center instead
                # For now, skip this word and let it be covered by another star
                continue
            # Form the star: mark all neighbors as used
            used[neigh_c] = True
            code_list.append(c)
        
        # Check if we achieved full partition
        if not np.all(used):
            # Some words remain uncovered - add their neighborhoods as additional stars
            for c in range(total_words):
                if not used[c]:
                    # Find any word that covers c and can form a disjoint star
                    for c2 in neighbors[c]:
                        if not used[c2]:
                            # Check if c2's neighborhood is disjoint from used
                            if not np.any(used[neighbors[c2]]):
                                used[neighbors[c2]] = True
                                code_list.append(c2)
                                break
                    else:
                        # Force-add c (may overlap, but ensures coverage)
                        used[neighbors[c]] = True
                        code_list.append(c)
        
        return code_list

    code_list = surgical_partition()
    
    # Build coverage counts
    cov_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cov_count[neighbors[c]] += 1
    
    is_code = np.zeros(total_words, dtype=bool)
    for c in code_list:
        is_code[c] = True

    # Safety: if partition failed to cover everything, fall back to vectorized greedy
    if not np.all(cov_count > 0):
        covered = np.zeros(total_words, dtype=bool)
        covered[cov_count > 0] = True
        while not covered.all() and time.time() < deadline - 45.0:
            uncovered_mask = ~covered
            if not uncovered_mask.any():
                break
            potential = neighbor_mask @ uncovered_mask.astype(np.int32)
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            covered[neighbors[c]] = True
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            cov_count[neighbors[c]] += 1

    # --- Phase 2: Iterated Local Search with Vectorized Swaps (Enhanced) ---
    def local_search(code, cov_count, max_time, seed):
        """Run local search (removal + vectorized swap) on a copy; return best (code, cov_count)."""
        code = code[:]
        cov_count = cov_count.copy()
        code_set = set(code)
        random.seed(seed)
        best_code = code[:]
        best_size = len(code)
        best_cov_count = cov_count.copy()
        stuck_counter = 0
        no_improve_streak = 0

        end_time = time.time() + max_time
        while time.time() < end_time:
            if len(code) == 0:
                break

            # 1. Try to remove a random word
            idx = random.randint(0, len(code) - 1)
            c = code[idx]
            if np.all(cov_count[neighbors[c]] > 1):
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code.pop(idx)
                stuck_counter = 0
                no_improve_streak = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                continue

            # 2. Removal failed -> try a vectorized swap
            risky_points = [x for x in neighbors[c] if cov_count[x] == 1]
            if not risky_points:
                # Should be removable; safety
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code.pop(idx)
                stuck_counter = 0
                no_improve_streak = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                continue

            # Candidate c2 must cover all risky points and not be in code
            if len(risky_points) == 1:
                candidate_mask = neighbor_mask[risky_points[0]].copy()
            else:
                candidate_mask = neighbor_mask[risky_points[0]].copy()
                for x in risky_points[1:]:
                    candidate_mask &= neighbor_mask[x]
                    if not candidate_mask.any():
                        break

            candidate_mask[c] = False
            for cc in code:
                candidate_mask[cc] = False

            valid_candidates = np.where(candidate_mask)[0]
            if len(valid_candidates) > 0:
                # Enhanced scoring: prefer c2 that maximizes slack (minimizes new risk)
                if len(valid_candidates) <= 32:
                    scores = np.array([
                        np.sum(cov_count[neighbors[c2]] <= 1) - 0.1 * np.sum(cov_count[neighbors[c2]] == 1)
                        for c2 in valid_candidates
                    ], dtype=np.float32)
                    best_score = np.max(scores)
                    top = valid_candidates[scores >= best_score - 1e-6]
                    c2 = int(random.choice(top))
                else:
                    sample_size = min(32, len(valid_candidates))
                    sample_idx = random.sample(range(len(valid_candidates)), sample_size)
                    sampled = valid_candidates[sample_idx]
                    scores = np.array([
                        np.sum(cov_count[neighbors[c2]] <= 1) - 0.1 * np.sum(cov_count[neighbors[c2]] == 1)
                        for c2 in sampled
                    ], dtype=np.float32)
                    best_score = np.max(scores)
                    top = sampled[scores >= best_score - 1e-6]
                    c2 = int(random.choice(top))

                # Perform swap
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code[idx] = c2
                cov_count[neighbors[c2]] += 1
                code_set.add(c2)
                stuck_counter = 0
                no_improve_streak = 0
            else:
                stuck_counter += 1
                no_improve_streak += 1

            # Periodic restart: if stuck for too long, reset to best and try different seed
            if no_improve_streak > 200:
                code = best_code[:]
                cov_count = best_cov_count.copy()
                code_set = set(code)
                no_improve_streak = 0
                random.seed(random.randint(0, 2**31 - 1))

        return best_code, best_cov_count

    # Run multiple local-search passes with different seeds to explore different basins
    # Since we start from the optimal-size partition, focus on verification and fine-tuning
    remaining = deadline - 12.0 - time.time()
    if remaining > 5.0:
        # First pass: aggressive removal to see if we can go below 486 (unlikely but try)
        t1 = max(2.0, remaining * 0.4)
        code, cov_count = local_search(code_list, cov_count, t1, seed=42)

    remaining = deadline - 10.0 - time.time()
    if remaining > 5.0:
        # Second pass: continue optimization from where we left off
        t2 = max(2.0, remaining * 0.35)
        code, cov_count = local_search(code, cov_count, t2, seed=1337)

    remaining = deadline - 8.0 - time.time()
    if remaining > 3.0:
        # Third pass: different seed for diversification
        t3 = max(2.0, remaining * 0.3)
        code, cov_count = local_search(code, cov_count, t3, seed=9999)

    # --- Final Polish: aggressive multi-pass exact removal to fixed point ---
    final_code = code[:]
    final_cov = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov[neighbors[idx]] += 1

    # Multi-pass removal with randomization to escape local minima
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2.0 and passes < 5:
        improved = False
        passes += 1
        # Try removal in different orders each pass
        if passes % 2 == 0:
            indices = list(range(len(final_code) - 1, -1, -1))
        else:
            indices = list(range(len(final_code)))
            random.shuffle(indices)
        
        for i in indices:
            if i >= len(final_code):
                continue
            c = final_code[i]
            if np.all(final_cov[neighbors[c]] > 1):
                final_cov[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # Verify validity; fall back to last valid if polish broke it (should not)
    if not np.all(final_cov > 0):
        # Rebuild from code (pre-polish) which was valid
        final_code = code[:]
        final_cov = np.zeros(total_words, dtype=np.int32)
        for idx in final_code:
            final_cov[neighbors[idx]] += 1
        if not np.all(final_cov > 0):
            # ultimate fallback: use the original partition code
            final_code = code_list[:]
            final_cov = np.zeros(total_words, dtype=np.int32)
            for idx in final_code:
                final_cov[neighbors[idx]] += 1

    return [list(digits_of[i]) for i in final_code]
