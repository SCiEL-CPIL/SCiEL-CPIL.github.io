"""Optimized ternary covering code K_3(7,1) using greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1)."""
    N = 2187
    powers = [1, 3, 9, 27, 81, 243, 729]

    # Precompute covers as tuples of indices within Hamming distance 1
    covers = [None] * N
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        indices = [i]
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    new_i = i + (val - digits[pos]) * powers[pos]
                    indices.append(new_i)
        covers[i] = tuple(indices)

    # Build adjacency matrix: adj[i,j]=1 if j is within distance 1 of i
    adj = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in covers[i]:
            adj[i, j] = 1

    def greedy_cover(randomize=False, seed=None):
        """Greedy set cover using vectorized scoring."""
        if seed is not None:
            random.seed(seed)

        uncovered = np.ones(N, dtype=np.uint8)
        selected = []
        remaining = N

        while remaining > 0:
            # Vectorized score: how many uncovered words each candidate covers
            scores = adj.dot(uncovered)
            best_count = int(scores.max())
            if best_count == 0:
                break

            best_mask = (scores == best_count)
            best_indices = np.where(best_mask)[0]

            if randomize and len(best_indices) > 1:
                idx = int(random.choice(best_indices))
            else:
                idx = int(best_indices[0])

            selected.append(idx)
            uncovered[covers[idx]] = 0
            remaining -= best_count

            if time.time() > deadline - 5.0:
                break

        return selected, remaining

    def local_search(code, deadline):
        """Local search: removal, swap, and 2-for-1 moves."""
        if not code:
            return code

        code_set = set(code)
        ci = covers
        N_local = N

        # Coverage counts
        counts = np.zeros(N_local, dtype=np.int32)
        for c in code_set:
            counts[ci[c]] += 1

        best_code = list(code_set)
        best_size = len(best_code)
        improved = True

        while improved and time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Removals
            current_list = list(code_set)
            random.shuffle(current_list)
            for c in current_list:
                if c not in code_set:
                    continue
                if np.all(counts[ci[c]] > 1):
                    counts[ci[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

            # Phase 2: Swaps
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if code_list:
                    for _ in range(100):
                        if time.time() > deadline - 5.0:
                            break
                        c_old = random.choice(code_list)
                        if c_old not in code_set:
                            continue
                        c_new = random.randint(0, N_local - 1)
                        if c_new in code_set:
                            continue
                        # Check if swap is valid
                        old_cover = np.array(ci[c_old], dtype=np.int32)
                        new_cover = np.array(ci[c_new], dtype=np.int32)
                        # Words only covered by c_old must be covered by c_new
                        only_old = (counts[old_cover] == 1)
                        if np.any(only_old):
                            if not np.all(new_cover[np.where(only_old)[0]] == 1):
                                continue
                        counts[old_cover] -= 1
                        counts[new_cover] += 1
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        improved = True
                        code_list = list(code_set)

            # Phase 3: 2-for-1 moves
            if time.time() < deadline - 5.0:
                code_list = list(code_set)
                if len(code_list) >= 2:
                    for _ in range(50):
                        if time.time() > deadline - 5.0:
                            break
                        c1, c2 = random.sample(code_list, 2)
                        if c1 not in code_set or c2 not in code_set:
                            continue

                        ci1 = np.array(ci[c1], dtype=np.int32)
                        ci2 = np.array(ci[c2], dtype=np.int32)

                        # Words exclusively covered by c1 or c2
                        ex1 = (counts[ci1] == 1)
                        ex2 = (counts[ci2] == 1)
                        # Union of exclusive words
                        exclusive_mask = np.zeros(N_local, dtype=bool)
                        exclusive_mask[ci1[ex1]] = True
                        exclusive_mask[ci2[ex2]] = True
                        exclusive_words = np.where(exclusive_mask)[0]

                        if len(exclusive_words) == 0:
                            continue

                        # Sample candidates
                        sample_size = min(150, N_local)
                        candidates = random.sample(range(N_local), sample_size)
                        best_new = -1
                        best_cov = -1
                        for c_new in candidates:
                            if c_new in code_set:
                                continue
                            cov_arr = np.array(ci[c_new], dtype=np.int32)
                            cov_count = int(np.sum(cov_arr[exclusive_words] == 1))
                            if cov_count > best_cov:
                                best_cov = cov_count
                                best_new = c_new

                        if best_new == -1 or best_cov < len(exclusive_words):
                            continue

                        c_new = best_new
                        ci_new = np.array(ci[c_new], dtype=np.int32)

                        # Full validation
                        valid = True
                        # Check words covered by c1
                        for j in ci1:
                            cnt = counts[j]
                            if j in ci2:
                                if cnt <= 2 and ci_new[j] != 1:
                                    valid = False
                                    break
                            else:
                                if cnt <= 1 and ci_new[j] != 1:
                                    valid = False
                                    break
                        if valid:
                            for j in ci2:
                                if j in ci1:
                                    continue
                                cnt = counts[j]
                                if cnt <= 1 and ci_new[j] != 1:
                                    valid = False
                                    break

                        if valid:
                            counts[ci1] -= 1
                            counts[ci2] -= 1
                            counts[ci_new] += 1
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(c_new)
                            improved = True
                            if len(code_set) < best_size:
                                best_size = len(code_set)
                                best_code = list(code_set)
                            code_list = list(code_set)

        return best_code

    best_code = None
    best_size = N

    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_cover(randomize=False)
    if unc == 0:
        code = local_search(code, deadline)
        if code:
            best_code = code
            best_size = len(code)

    # 2. Randomized restarts
    attempts = 0
    while time.time() < deadline - 8.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts * 7919)
        if unc == 0:
            code = local_search(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
        if best_size <= 186:
            break

    if best_code is None:
        code, unc = greedy_cover(randomize=False)
        if unc > 0:
            # Complete the cover
            uncovered_idx = np.where(np.ones(N, dtype=np.uint8))[0]
            for idx in uncovered_idx:
                code.append(idx)
        best_code = code
        best_size = len(best_code)

    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * 7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)

    return result


def _construct_general(q, n, radius, deadline):
    """General fallback."""
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)

    uncovered = set(range(total))
    selected = []

    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)

    if uncovered:
        for j in uncovered:
            selected.append(j)

    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
