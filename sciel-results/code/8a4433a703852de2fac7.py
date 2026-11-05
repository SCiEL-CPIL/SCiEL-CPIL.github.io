import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Pre-compute distance-2 neighbors (ball of radius 2) for swap moves
    neighbors2 = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs2 = set(neighbors[idx])
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for d1 in [1, 2]:
                    for d2 in [1, 2]:
                        new_word = word[:]
                        new_word[pos1] = (word[pos1] + d1) % q
                        new_word[pos2] = (word[pos2] + d2) % q
                        new_idx = sum(new_word[i] * powers[i] for i in range(n))
                        nbrs2.add(new_idx)
        neighbors2[idx] = list(nbrs2)

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed=None, deterministic=False):
        if deterministic:
            rng = None
        else:
            rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                if deterministic:
                    best_idx = int(candidates[0])
                else:
                    best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices

        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def peeling_pass(selected_list, coverage_count, deadline, max_rounds=50):
        """Remove most-redundant codewords iteratively (maximize min coverage in ball)."""
        selected_list = list(selected_list)
        coverage_count = coverage_count.copy()
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]

        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            if not selected_list:
                break

            # Find codeword with highest minimum coverage in its ball (most redundant)
            # min_coverage[i] = min of coverage_count[balls[i]] for i in selected
            min_covers = np.min(coverage_count[balls[selected_list]], axis=1)
            max_min = np.max(min_covers)
            if max_min <= 1:
                break  # No more removable codewords

            # Pick the one with highest min coverage, break ties by index
            candidates = np.where(min_covers == max_min)[0]
            if len(candidates) > 1:
                # Among ties, pick the one that frees up the most unique coverage
                best_i = int(candidates[0])
                best_unique = -1
                for ci in candidates[:20]:  # limit to 20 for speed
                    idx = selected_list[int(ci)]
                    unique_count = np.sum(coverage_count[balls[idx]] == 1)
                    if unique_count < best_unique or best_unique == -1:
                        best_unique = unique_count
                        best_i = int(ci)
            else:
                best_i = int(candidates[0])

            idx = selected_list[best_i]
            coverage_count[balls[idx]] -= 1
            selected_list.pop(best_i)
            selected_set.discard(idx)

            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]

        return best_code, rebuild_coverage(best_code)

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Peeling pass: remove most-redundant codewords
            peeled, coverage_count = peeling_pass(selected_list, coverage_count, deadline, max_rounds=30)
            if len(peeled) < len(selected_list):
                selected_list = peeled
                selected_set = set(selected_list)
            if len(selected_list) < best_len:
                best_len = len(selected_list)
                best_code = selected_list[:]

            if time.time() > deadline:
                break

            # 2. Neighbor-restricted swap: move a random codeword to a point in its 2-ball
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            progress = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                # Use distance-2 neighbors for richer move space
                nbrs = list(neighbors2[c])
                rng.shuffle(nbrs)
                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue
                    # Tentatively swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue
                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break
                if progress:
                    break

            if progress:
                # After a swap, prune to catch newly-redundant codewords
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """Final high-precision pass: peeling + swaps."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(7)

        # Peeling sweeps until fixed point
        for _ in range(50):
            if time.time() > deadline:
                break
            changed = False
            peeled, coverage_count = peeling_pass(selected_list, coverage_count, deadline, max_rounds=10)
            if len(peeled) < len(selected_list):
                selected_list = peeled
                selected_set = set(selected_list)
                changed = True
            if not changed:
                break

        # Neighbor-swap attempts to reduce size further
        for _ in range(300):
            if time.time() > deadline:
                break
            n_sel = len(selected_list)
            if n_sel == 0:
                break
            i = rng.randrange(n_sel)
            c = selected_list[i]
            nbrs = list(neighbors2[c])
            rng.shuffle(nbrs)
            for c_prime in nbrs:
                if c_prime == c or c_prime in selected_set:
                    continue
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1
                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    continue
                selected_list[i] = c_prime
                selected_set.discard(c)
                selected_set.add(c_prime)
                # Prune after swap
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                break

        return best_code

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Greedy seeds: deterministic primary + a couple random if time allows
    best_code = None
    best_len = float('inf')

    # Deterministic greedy (lexicographic tie-break) - primary seed
    selected = greedy_set_cover(deterministic=True)
    selected = prune(selected)
    best_len = len(selected)
    best_code = selected[:]

    # A few random-seed greedy runs if we have time
    for seed in range(3):
        if time.time() > deadline - 60:
            break
        selected = greedy_set_cover(seed=seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    # Local search walk (peeling + neighbor-restricted swaps)
    best_code = local_search_walk(best_code, deadline - 15.0)

    # Final polish pass
    best_code = polish(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
