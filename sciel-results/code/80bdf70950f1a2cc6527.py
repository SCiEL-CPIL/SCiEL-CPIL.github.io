"""Covering code K_3(6,1) - vectorized greedy set cover + local optimization."""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 108  # safety margin before 120s limit

    total = q ** n

    # --- Fast index conversions using precomputed powers ---
    powers = [q ** i for i in range(n)]

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    # --- Precompute balls as numpy arrays for all words ---
    # balls_flat[i] = array of indices covered by codeword i (size 1 + n*(q-1))
    ball_size = 1 + n * (q - 1)
    balls_flat = np.zeros((total, ball_size), dtype=np.int32)

    for w_idx in range(total):
        w = idx_to_word(w_idx)
        ball = [w_idx]
        for pos in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[pos] = (w[pos] + delta) % q
                ball.append(word_to_idx(new_w))
        balls_flat[w_idx, :] = ball

    # --- Coverage count approach ---
    # cov_count[x] = number of codewords whose ball covers word x
    # A set of codewords is valid iff min(cov_count) >= 1

    def make_greedy(seed_rng=None):
        """Greedy set cover: repeatedly pick codeword covering most uncovered words."""
        if seed_rng is None:
            seed_rng = random.Random()
        cov_count = np.zeros(total, dtype=np.int32)
        selected = []
        # uncovered_count = total - sum(cov_count > 0)
        # Track uncovered words
        uncovered = np.ones(total, dtype=bool)
        remaining = total

        while remaining > 0:
            if time.time() > deadline:
                break
            # Find codeword covering most uncovered words
            # For each candidate, count how many uncovered words are in its ball
            # Vectorized: for all candidates, sum uncovered[balls_flat[candidate]]
            # This is total * ball_size operations per iteration - expensive for large total
            # Instead, use incremental: maintain uncovered set, for each candidate compute overlap

            # Efficient approach: for each uncovered word, which candidates cover it?
            # Precompute reverse: for each word x, list of codewords covering it
            # But that's total * ball_size memory. For 729*13 ~ 9477, fine.

            # Alternative: just iterate over candidates and count uncovered in ball
            # 729 candidates * 13 ball entries = ~9477 ops per iteration, very fast
            best_cw = -1
            best_count = 0
            for cw in range(total):
                cnt = np.sum(uncovered[balls_flat[cw]])
                if cnt > best_count:
                    best_count = cnt
                    best_cw = cw
                elif cnt == best_count and cnt > 0:
                    # Tie-break: prefer codeword with lower index for determinism
                    pass

            if best_cw < 0 or best_count == 0:
                break

            selected.append(best_cw)
            # Update coverage
            ball = balls_flat[best_cw]
            for pos in ball:
                cov_count[pos] += 1
            new_uncovered = ball[uncovered[ball] == False]  # no, just mark covered
            uncovered[ball] = False
            remaining -= best_count

        return selected, cov_count

    def is_redundant_fast(cw_idx, cov_count):
        """Check if removing cw_idx leaves all words in its ball still covered (count >= 2 before removal)."""
        ball = balls_flat[cw_idx]
        return np.all(cov_count[ball] >= 2)

    def try_remove_redundant(codeword_set, cov_count):
        """Remove all redundant codewords. Returns (new_set, new_cov_count, removed_count)."""
        changed = True
        removed = 0
        while changed:
            changed = False
            for cw in list(codeword_set):
                if time.time() > deadline:
                    break
                if is_redundant_fast(cw, cov_count):
                    ball = balls_flat[cw]
                    cov_count[ball] -= 1
                    codeword_set.discard(cw)
                    changed = True
                    removed += 1
        return codeword_set, cov_count, removed

    def try_swap(old_cw, new_cw, codeword_set, cov_count):
        """Try swapping old_cw -> new_cw. Returns (success, new_cov_count)."""
        # Check: after removing old_cw, are all words still covered?
        # After adding new_cw, are all words still covered?
        # Remove old: cov_count[ball(old)] -= 1
        # Add new: cov_count[ball(new)] += 1
        # Valid iff min(cov_count) >= 1 after both changes

        ball_old = balls_flat[old_cw]
        ball_new = balls_flat[new_cw]

        # Tentatively apply
        cov_count[ball_old] -= 1
        cov_count[ball_new] += 1

        if np.min(cov_count) >= 1:
            codeword_set.discard(old_cw)
            codeword_set.add(new_cw)
            return True, cov_count
        else:
            # Revert
            cov_count[ball_old] += 1
            cov_count[ball_new] -= 1
            return False, cov_count

    def local_optimize(codeword_set, cov_count):
        """Aggressive local optimization: remove redundant, then try targeted swaps."""
        codeword_set, cov_count, _ = try_remove_redundant(codeword_set, cov_count)
        size = len(codeword_set)

        # Identify words with coverage count == 1 (critical words)
        # A codeword is "critical" if it's the only cover for some word
        # Swapping a critical codeword to a position that covers more critical words
        # can enable further removals

        # Build: for each codeword, which critical words does it cover?
        # critical_words = words where cov_count == 1
        critical_words = np.where(cov_count == 1)[0]
        
        if len(critical_words) > 0:
            # For each codeword in the set, count how many critical words it covers
            cw_list = list(codeword_set)
            cw_critical_counts = np.zeros(len(cw_list), dtype=np.int32)
            for i, cw in enumerate(cw_list):
                ball = balls_flat[cw]
                cw_critical_counts[i] = np.sum(cov_count[ball] == 1)
            
            # Sort codewords by critical count (descending) - most critical first
            critical_order = np.argsort(cw_critical_counts)[::-1]

            # Try swaps: replace high-critical codewords with random positions
            # that might cover different critical words
            for it in range(300):
                if time.time() > deadline:
                    break
                if size == 0:
                    break
                
                # Pick a codeword that covers many critical words (good candidate to replace)
                if len(critical_order) > 0 and critical_order[0] < len(cw_list):
                    old_cw = cw_list[int(critical_order[0])]
                else:
                    old_cw = random.choice(cw_list)
                
                new_cw = random.randint(0, total - 1)
                if new_cw in codeword_set:
                    continue

                success, cov_count = try_swap(old_cw, new_cw, codeword_set, cov_count)
                if success:
                    # Try to remove redundant after swap
                    codeword_set, cov_count, rem = try_remove_redundant(codeword_set, cov_count)
                    size = len(codeword_set)
                    
                    # Rebuild critical info if set changed
                    if rem > 0:
                        critical_words = np.where(cov_count == 1)[0]
                        cw_list = list(codeword_set)
                        cw_critical_counts = np.zeros(len(cw_list), dtype=np.int32)
                        for i, cw in enumerate(cw_list):
                            ball = balls_flat[cw]
                            cw_critical_counts[i] = np.sum(cov_count[ball] == 1)
                        critical_order = np.argsort(cw_critical_counts)[::-1]
        
        # Additional random swaps for diversity
        for it in range(100):
            if time.time() > deadline:
                break
            if size == 0:
                break
            cw_list = list(codeword_set)
            old_cw = random.choice(cw_list)
            new_cw = random.randint(0, total - 1)
            if new_cw in codeword_set:
                continue

            success, cov_count = try_swap(old_cw, new_cw, codeword_set, cov_count)
            if success:
                codeword_set, cov_count, rem = try_remove_redundant(codeword_set, cov_count)
                size = len(codeword_set)

        return codeword_set, cov_count

    # --- Main: multiple greedy restarts with different seeds ---
    best_size = total + 1
    best_set = None
    best_cov = None

    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Different greedy strategies per restart
        if r == 0:
            # Standard greedy
            selected, cov_count = make_greedy()
        else:
            # Perturbed greedy: use random tie-breaking and occasional non-greedy picks
            rng = random.Random(r * 12345 + 67890)
            cov_count = np.zeros(total, dtype=np.int32)
            uncovered = np.ones(total, dtype=bool)
            remaining = total
            selected = []

            while remaining > 0:
                if time.time() > deadline:
                    break
                # Find top-5 candidates by coverage of uncovered
                scores = np.zeros(total, dtype=np.int32)
                for cw in range(total):
                    scores[cw] = np.sum(uncovered[balls_flat[cw]])

                # Get top candidates
                top_k = min(5, total)
                top_indices = np.argsort(scores)[::-1][:top_k]

                # Randomly pick from top-k (with bias toward best)
                if rng.random() < 0.7:
                    best_cw = int(top_indices[0])
                else:
                    best_cw = int(top_indices[rng.randint(0, top_k - 1)])

                if scores[best_cw] == 0:
                    break

                selected.append(best_cw)
                ball = balls_flat[best_cw]
                cov_count[ball] += 1
                uncovered[ball] = False
                remaining -= scores[best_cw]

        if len(selected) == 0:
            continue

        codeword_set = set(selected)
        # Rebuild cov_count from scratch to be safe
        cov_count = np.zeros(total, dtype=np.int32)
        for cw in codeword_set:
            cov_count[balls_flat[cw]] += 1

        # Local optimization
        codeword_set, cov_count = local_optimize(codeword_set, cov_count)

        if len(codeword_set) < best_size:
            best_size = len(codeword_set)
            best_set = set(codeword_set)
            best_cov = cov_count.copy()

            # If we hit the known optimum, stop early
            if best_size <= 73:
                break

    # --- Final: if we have a good result, do multiple rounds of optimization ---
    if best_set is not None:
        # Do several polish passes with fresh randomization
        polish_passes = 3
        for p in range(polish_passes):
            if time.time() > deadline - 5:
                break
            # Re-seed random for different swap patterns
            random.seed(p * 9999 + 42)
            final_set, final_cov = local_optimize(set(best_set), best_cov.copy())
            if len(final_set) < best_size:
                best_size = len(final_set)
                best_set = final_set
                best_cov = final_cov
            else:
                # No improvement, break early
                break

    # --- Fallback: tetracode construction ---
    if best_set is None or best_size > 81:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        initial_cw = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                cw = h + list(tail)
                initial_cw.append(word_to_idx(cw))
        best_set = set(initial_cw)
        best_size = len(best_set)

    # --- Convert to result ---
    result = [idx_to_word(cw) for cw in best_set]

    # --- Verify validity ---
    verify_cov = np.zeros(total, dtype=np.int32)
    for cw in result:
        cw_idx = word_to_idx(cw)
        verify_cov[balls_flat[cw_idx]] += 1

    if np.min(verify_cov) < 1:
        # Fallback to tetracode
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                result.append(h + list(tail))

    return result
