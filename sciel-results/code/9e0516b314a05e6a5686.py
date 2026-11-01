"""Combined program: fast greedy init + vectorized local search (deletion + swap) for K_3(7,1)."""
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187
    ball_size = 1 + 7 * 2  # 15

    # Precompute powers of 3
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = []
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        ball = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    ball.append(new_i)
        ball_indices.append(ball)

    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]

    # Precompute covers as bitsets (Python ints) for fast uncovered tracking
    covers = [0] * N
    for i in range(N):
        mask = 1 << i
        for j in ball_indices[i]:
            if j != i:
                mask |= (1 << j)
        covers[i] = mask

    full_mask = (1 << N) - 1

    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        uncovered = full_mask
        selected = []

        while uncovered:
            best_count = -1
            best_idx = -1
            best_indices = []

            # Optimization: check candidates in random order for randomized version
            candidates = list(range(N))
            if seed is not None:
                rng.shuffle(candidates)

            for i in candidates:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_indices = [i]
                elif cnt == best_count:
                    best_indices.append(i)

            if best_count == 0:
                break

            if len(best_indices) > 1 and seed is not None:
                idx = rng.choice(best_indices)
            else:
                idx = best_idx

            selected.append(idx)
            uncovered &= ~covers[idx]

            if time.time() > deadline - 5.0:
                break

        return selected, uncovered

    # Delete phase: remove redundant codewords
    def run_delete_phase(code_list, cnt_arr):
        """Removes redundant codewords. Returns list of removed indices."""
        removed = []
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword in random order
            indices_to_try = list(range(len(code_list)))
            random.shuffle(indices_to_try)

            for idx in indices_to_try:
                if idx >= len(code_list):
                    continue
                c = code_list[idx]
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    # Safe to remove
                    cnt_arr[ball] -= 1
                    code_list.pop(idx)
                    removed.append(c)
                    changed = True
        return removed

    # Get valid swaps for removing codeword c
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c."""
        ball = ball_indices_np[c]
        unique = ball[cnt_arr[ball] == 1]

        if len(unique) == 0:
            # Any word not in code_set is valid
            return None

        # Intersection of balls of all unique words
        valid_set = set(ball_indices_np[unique[0]].tolist())
        for u in unique[1:]:
            valid_set &= set(ball_indices_np[u].tolist())
            if len(valid_set) == 0:
                return []

        # Filter out words already in code_set
        valid_list = [v for v in valid_set if v not in code_set]
        return valid_list

    # Main search
    best_code = None
    best_size = N

    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_build(seed=None)
    if not unc:
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball_indices_np[c]] += 1
        run_delete_phase(code, cnt)
        best_code = code
        best_size = len(code)

    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_build(seed=attempts)
        if not unc:
            cnt = np.zeros(N, dtype=np.int32)
            for c in code:
                cnt[ball_indices_np[c]] += 1

            # Local search loop
            no_improve = 0
            while time.time() < deadline - 5.0 and no_improve < 20:
                # Delete phase
                removed = run_delete_phase(code, cnt)
                if removed:
                    no_improve = 0
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                else:
                    no_improve += 1

                # Swap phase
                if time.time() > deadline - 5.0:
                    break

                code_set = set(code)
                swaps_attempted = 0
                max_swaps = 100

                while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
                    swaps_attempted += 1
                    if not code:
                        break

                    # Pick random codeword to remove
                    c = random.choice(code)
                    valid_swaps = get_valid_swaps(c, code_set, cnt)

                    if valid_swaps is None:
                        # Any non-code word works
                        c_prime = random.randint(0, N - 1)
                        while c_prime in code_set:
                            c_prime = random.randint(0, N - 1)
                    elif len(valid_swaps) == 0:
                        continue
                    else:
                        c_prime = random.choice(valid_swaps)

                    # Perform swap
                    code.remove(c)
                    code.append(c_prime)
                    code_set.remove(c)
                    code_set.add(c_prime)
                    cnt[ball_indices_np[c]] -= 1
                    cnt[ball_indices_np[c_prime]] += 1

                    # After swap, run delete phase
                    removed = run_delete_phase(code, cnt)
                    if removed:
                        no_improve = 0
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                    else:
                        no_improve += 1

        if best_size <= 186:
            break

    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_build(seed=None)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
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
    """General fallback for other q, n, radius."""
    import itertools

    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)

    # Simple greedy
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

    # Ensure validity
    if uncovered:
        for j in uncovered:
            selected.append(j)

    return [list(all_words[i]) for i in selected]
