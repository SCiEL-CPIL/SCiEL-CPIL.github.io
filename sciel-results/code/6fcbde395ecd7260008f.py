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

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Remove any removable codeword (full sweep, deterministic)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # 2. Neighbor-restricted swap: move a random codeword to a point in its own ball
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            progress = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
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
        """Final high-precision pass: full deterministic sweeps + a few neighbor swaps."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(7)

        # Deterministic removal sweeps until fixed point
        for _ in range(50):
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break
            if not changed:
                break

        # A few neighbor-swap attempts to reduce size further
        for _ in range(200):
            if time.time() > deadline:
                break
            n_sel = len(selected_list)
            if n_sel == 0:
                break
            i = rng.randrange(n_sel)
            c = selected_list[i]
            nbrs = list(neighbors[c])
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
    deadline = start_time + 110.0

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

    # Local search walk (neighbor-restricted swaps + prune)
    best_code = local_search_walk(best_code, deadline - 10.0)

    # Final polish pass
    best_code = polish(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
