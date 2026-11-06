import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (evaluator kills at 120s usually, but contract says 120s, so 115 is safe)

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls (neighbors including self)
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        # Precompute digit arrays for all words
        digit_arrays = [int_to_base3(i, n) for i in range(N)]
        
        # Precompute full distance matrix using vectorized numpy operations
        # Convert digit_arrays to a numpy array of shape (N, n)
        digits_np = np.array(digit_arrays, dtype=np.uint8)  # (N, n)
        
        # dist_matrix[i,j] = number of positions where digits_np[i] != digits_np[j]
        # = n - sum over k of (digits_np[i,k] == digits_np[j,k])
        # We can compute this as: for each position k, compute the equality matrix, sum, then n - sum
        # Equality matrix for position k: (N, N) boolean, True where digits_np[:, k] equal
        # Sum over k gives (N, N) array of number of matching positions
        # dist = n - matching_count
        
        # This uses N*N*n memory temporarily if done naively, but we can do it position by position
        matching_count = np.zeros((N, N), dtype=np.int32)
        for k in range(n):
            col = digits_np[:, k]  # (N,)
            # Equality matrix: col[:, None] == col[None, :]
            eq = (col[:, None] == col[None, :])  # (N, N) boolean
            matching_count += eq.astype(np.int32)
        
        dist_matrix = (n - matching_count).astype(np.uint8)
        np.fill_diagonal(dist_matrix, 0)
        
        # Free memory
        del digits_np
        del matching_count

        def greedy_cover(seed, start_code=None):
            """
            Greedy set cover with conflict-aware scoring.
            Score = marginal_coverage - lambda * average_overlap_with_selected
            where overlap measures how much the candidate's ball overlaps
            with balls of already-selected codewords (proxy for low pairwise distance).
            """
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            in_code = [False] * N
            selected_set = set()
            
            # Track overlap: for each candidate c, overlap_score[c] = sum of 
            # "closeness" to selected codewords. We use (n - dist) as closeness,
            # so higher overlap means closer to existing codewords (more redundant).
            # We want to penalize high overlap.
            # overlap_penalty[c] = sum over s in selected of (n - dist_matrix[c][s]) / n
            # This is in [0, |selected|]. We normalize by dividing by |selected| for average.
            overlap_penalty = np.zeros(N, dtype=np.float32)
            num_selected = 0

            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    selected_set.add(c)
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1
                num_selected = len(start_code)
                # Compute initial overlap penalties
                for c in start_code:
                    # Add contribution of c to overlap_penalty of all other candidates
                    # closeness = n - dist_matrix[., c]
                    closeness = (n - dist_matrix[:, c]).astype(np.float32)
                    overlap_penalty += closeness / float(n)

            # We'll use a simple re-scoring approach: at each step, compute score for all
            # uncovered candidates and pick the best. This is O(N^2) per step but N=2187
            # and we have ~186 steps, so ~70M operations. Might be slow in Python.
            # Alternative: use heap with lazy updates, but the scoring is more complex.
            # Let's try a batch approach: recompute scores every few steps.
            
            # Actually, let's use a simpler heuristic: 
            # score[c] = cnt[c] - lambda * overlap_penalty[c]
            # where lambda is a tunable parameter.
            # We'll use a heap with periodic rescoring.
            
            lambda_overlap = 0.15  # penalty weight for overlap/closeness
            
            # Build initial heap
            # score = cnt - lambda * overlap_penalty
            # We want max score, so heap key = -score
            heap = []
            for i in range(N):
                if not in_code[i]:
                    score = cnt[i] - lambda_overlap * overlap_penalty[i]
                    heap.append((-score, i))
            heapq.heapify(heap)

            # Track which candidates have been scored with current overlap_penalty
            # to avoid stale entries. We'll use a version counter.
            # Simpler: just pop and check if still valid (not in_code) and rescore if needed.
            # Since overlap_penalty changes every step, heap entries become stale quickly.
            # Let's just recompute the best candidate from scratch each step.
            # N=2187, steps~186, so 186*2187 = ~406K evaluations. Each eval is O(1) with precomputed arrays.
            
            while uncovered_count > 0:
                if time.time() > deadline - 10:
                    break
                    
                # Find best candidate by scanning all not-in-code candidates
                best_c = -1
                best_score = -1e9
                for c in range(N):
                    if in_code[c]:
                        continue
                    if cnt[c] == 0:
                        continue
                    score = cnt[c] - lambda_overlap * overlap_penalty[c]
                    # Add small random tiebreaker for diversity
                    if score > best_score + 1e-6:
                        best_score = score
                        best_c = c
                    elif abs(score - best_score) <= 1e-6 and rng.random() < 0.1:
                        best_c = c

                if best_c < 0:
                    break

                c = best_c
                code.append(c)
                in_code[c] = True
                selected_set.add(c)
                num_selected += 1

                # Update overlap_penalty: add contribution of new codeword c
                closeness = (n - dist_matrix[:, c]).astype(np.float32)
                overlap_penalty += closeness / float(n)

                # Update coverage
                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """
            Local search with deletion and 1-swap moves.
            Optimized for speed.
            """
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                for i in indices:
                    if time.time() > time_limit:
                        break
                    c_remove = code_list[i]

                    # Try deletion first
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # Try 1-swap: find a single codeword covering all newly_uncovered
                    found = -1
                    # Heuristic: sort critical words by rarity (fewest coverers)
                    crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                    cand_pool = None
                    for w in crit_sorted:
                        candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                        if cand_pool is None:
                            cand_pool = set(candidates_w)
                        else:
                            cand_pool.intersection_update(candidates_w)
                        if not cand_pool:
                            break
                    
                    if cand_pool:
                        # Pick a candidate that minimizes "waste" or maximizes redundancy?
                        # For now, just pick one that works. 
                        # To be stochastic, we could shuffle cand_pool.
                        # But finding *any* valid swap is good progress.
                        # Let's pick the one that covers the most *other* uncovered words? 
                        # No, just the first valid one is fine for deterministic local search.
                        # To add stochasticity, shuffle the list before iterating?
                        # The set iteration order is arbitrary. 
                        # Let's convert to list and shuffle.
                        cand_list = list(cand_pool)
                        rng.shuffle(cand_list)
                        for c_add in cand_list:
                            bs = ball_sets[c_add]
                            if all(w in bs for w in newly_uncovered):
                                found = c_add
                                break

                    if found >= 0:
                        code_list[i] = found
                        in_code[c_remove] = False
                        in_code[found] = True
                        for w in ball_indices[found]:
                            cover_count[w] += 1
                        improved = True
                        break

                    # Restore if no improvement
                    for w in ball_indices[c_remove]:
                        cover_count[w] += 1

            return code_list

        def run_greedy_local_search(seed):
            code = greedy_cover(seed)
            code = refine(code)
            time_left = deadline - 5
            code = local_search(code, seed, time_left)
            
            # Final validity check
            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if min(cover_count) < 1:
                return None
            return code

        def run_perturbation(seed, base_code, time_limit):
            """
            Take base_code, remove a random subset, and re-greedy the rest.
            """
            if len(base_code) < 10:
                return None
            
            rng = random.Random(seed)
            # Remove 5-10% of codewords
            removal_fraction = 0.05 + 0.05 * rng.random()
            num_remove = max(1, int(len(base_code) * removal_fraction))
            
            code_list = list(base_code)
            rng.shuffle(code_list)
            to_remove = set(code_list[:num_remove])
            
            remaining_code = [c for c in code_list if c not in to_remove]
            
            new_code = greedy_cover(seed, start_code=remaining_code)
            
            # Refine
            new_code = refine(new_code)
            
            # Local Search
            new_code = local_search(new_code, seed + 1, time_limit)
            
            # Validity check
            cover_count = [0] * N
            for c in new_code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if min(cover_count) < 1:
                return None
                
            return new_code

        best_code = None
        best_size = float('inf')
        
        # Phase 1: Standard Multi-Start Greedy
        num_restarts = 0
        # We have ~115s. 
        # Greedy is fast. Local search is the bottleneck.
        # Let's do a few quick restarts first.
        
        while time.time() < deadline - 30:
            seed = num_restarts * 1000 + 42
            current_code = run_greedy_local_search(seed)
            num_restarts += 1
            
            if current_code is not None:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
            
            # If we find a very good code (e.g., < 190), we might want to spend more time refining it
            # via perturbation.
            if num_restarts >= 5:
                if time.time() > deadline - 40:
                    break

        # Phase 2: Perturbation / Kick Strategy on the best code found so far
        if best_code is not None:
            # We have some time left. Let's try to improve best_code.
            # We will perform multiple perturbation steps.
            num_perturbations = 0
            # Allocate remaining time to perturbations
            while time.time() < deadline - 5:
                seed = num_perturbations * 10000 + 999
                # Pass a tighter time limit to local search inside run_perturbation
                # Actually, run_perturbation uses the global deadline for local search.
                # We should be careful.
                
                perturbed_code = run_perturbation(seed, best_code, deadline - 5)
                num_perturbations += 1
                
                if perturbed_code is not None:
                    if len(perturbed_code) < best_size:
                        best_size = len(perturbed_code)
                        best_code = list(perturbed_code)
                        # If we improved, we can try more perturbations on the new best
                        # But if we didn't, we might be stuck. 
                        # To avoid getting stuck in a loop of non-improving perturbations,
                        # we could increase the perturbation strength or change seed strategy.
                    else:
                        # If no improvement, we might want to try a stronger perturbation
                        # or just stop if we've tried many.
                        if num_perturbations > 50:
                            break
        else:
            # Fallback if Phase 1 failed (shouldn't happen)
            best_code = run_greedy_local_search(0)
            if best_code is None:
                return []

        # Final Prune on the best code
        best_code = refine(best_code)
        
        # High-precision polish: multiple passes of local search with tighter time budget
        if time.time() < deadline - 10 and best_code:
            # Do 2-3 additional local search passes with fresh seeds
            for polish_iter in range(3):
                if time.time() > deadline - 10:
                    break
                polish_seed = 90000 + polish_iter * 7777
                polished = local_search(best_code, polish_seed, deadline - 10)
                if polished is not None:
                    polished = refine(polished)
                    # Validate
                    cover_count = [0] * N
                    for c in polished:
                        for w in balls[c]:
                            cover_count[w] += 1
                    if min(cover_count) >= 1 and len(polished) < len(best_code):
                        best_code = polished
        
        # Return as list of lists
        words = [int_to_base3(i, n) for i in best_code]
        return words

    # Fallback for other cases
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
    
    def get_counts(code_list):
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        flat_neighbors = np.concatenate([neighbors[i] for i in code_list])
        np.add.at(counts, flat_neighbors, 1)
        return counts

    def prune_code(code_list):
        counts = get_counts(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                if np.min(counts[neighbors[c]]) > 1:
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list

    def improve_by_swaps(code_list, max_iterations=100):
        counts = get_counts(code_list)
        code_set = set(code_list)
        improved = True
        iterations = 0
        
        neighbor_sets = [set(neigh) for neigh in neighbors]
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                
                crit_points = []
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        crit_points.append(nb)
                        
                if not crit_points:
                    if np.min(counts[neighbors[c]]) > 1:
                        code_list.remove(c)
                        code_set.remove(c)
                        for nb in neighbors[c]:
                            counts[nb] -= 1
                        improved = True
                        break
                    continue
                
                p = crit_points[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = 0
                    for nb in neighbors[w]:
                        if counts[nb] == 2:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    code_list.remove(c)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                        
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                        
                    improved = True
                    break
                    
        return code_list

    def run_greedy_local_search(seed):
        random.seed(seed)
        code = []
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        
        neighbor_array = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
        for i, neigh in enumerate(neighbors):
            neighbor_array[i, :] = neigh
        
        while uncovered_count > 0 and time.time() < deadline - 20:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            
            new_uncovered = neighbors[best_idx]
            newly_covered = 0
            for nb in new_uncovered:
                if uncovered[nb]:
                    uncovered[nb] = False
                    newly_covered += 1
            uncovered_count -= newly_covered
            
        code = prune_code(code)
        code = improve_by_swaps(code)
        
        counts = get_counts(code)
        if np.min(counts) < 1:
            return None
            
        return code

    best_code = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 1000 + 42
        current_code = run_greedy_local_search(seed)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        if num_restarts >= 10:
             if time.time() > deadline - 30:
                 break

    if best_code is None:
        best_code = run_greedy_local_search(0)
        if best_code is None:
             return []

    best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]
