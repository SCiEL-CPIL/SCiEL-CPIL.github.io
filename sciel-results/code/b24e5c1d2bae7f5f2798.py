import time
import numpy as np
import random


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 113.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]

    # Bitmask representation for O(1) ball union/intersection
    # total=2187 -> 44 uint64 words
    nbits = (total + 63) // 64
    ball_mask = np.zeros((total, nbits), dtype=np.uint64)
    for i in range(total):
        for idx in ball_indices_np[i]:
            ball_mask[i, idx >> 6] |= (np.uint64(1) << np.uint64(idx & 63))

    # Greedy set-cover construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        covered = np.zeros(total, dtype=bool)
        code = []

        while not np.all(covered):
            # Vectorized gain computation over all candidates
            gains = np.zeros(total, dtype=np.int32)
            for i in range(total):
                gains[i] = np.sum(~covered[ball_indices_np[i]])
            # break ties randomly among the max
            max_g = gains.max()
            if max_g == 0:
                break
            candidates = np.where(gains == max_g)[0]
            best_c = int(rng.choice(candidates.tolist()))
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code

    # Multiple greedy restarts to find good starting points
    best_initial = None
    best_initial_size = float('inf')

    init_code = greedy_build(seed=0)
    best_initial = set(init_code)
    best_initial_size = len(init_code)

    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 45.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)

    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)

    # Initialize coverage counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1

    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    code_set.remove(c)
                    cnt_arr[ball] -= 1
                    removed += 1
                    changed = True
        return removed

    def compute_unique_words(c, cnt_arr):
        """Return list of words in ball(c) with cnt==1."""
        ball = ball_indices_np[c]
        return ball[cnt_arr[ball] == 1]

    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c (keeps coverage)."""
        unique = compute_unique_words(c, cnt_arr)
        if len(unique) == 0:
            return None  # Any non-code word works
        # Intersection of balls of all unique words
        valid_set = set(ball_indices_np[unique[0]].tolist())
        for u in unique[1:]:
            valid_set &= set(ball_indices_np[u].tolist())
            if len(valid_set) == 0:
                return []
        valid_list = [v for v in valid_set if v not in code_set]
        return valid_list

    # Repair: greedily add codewords to restore full coverage
    def repair(code_set, cnt_arr):
        added = 0
        while cnt_arr.min() < 1:
            if time.time() > deadline - 3.0:
                break
            uncovered = np.where(cnt_arr == 0)[0]
            # pick candidate covering most uncovered
            best_c = -1
            best_gain = -1
            sample = uncovered[:min(len(uncovered), 50)]
            for u in sample:
                for cand in ball_indices_np[u]:
                    if cand in code_set:
                        continue
                    gain = np.sum(cnt_arr[ball_indices_np[cand]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_c = cand
            if best_c < 0:
                break
            code_set.add(best_c)
            cnt_arr[ball_indices_np[best_c]] += 1
            added += 1
        return added

    # Local Search Loop with SA-style probabilistic violating moves
    no_improve_counter = 0
    max_no_improve = 12
    # SA temperature schedule
    T = 1.0
    T_min = 0.05
    T_decay = 0.985

    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1

        # 2. Swap Phase
        swaps_attempted = 0
        max_swaps = 400
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break

            c = random.choice(list(current_code))
            valid_swaps = get_valid_swaps(c, current_code, cnt)

            if valid_swaps is None:
                c_prime = random.randint(0, total - 1)
                while c_prime in current_code:
                    c_prime = random.randint(0, total - 1)
                # valid (no unique words)
                do_swap = True
            elif len(valid_swaps) == 0:
                # No valid swap. SA: probabilistically accept a violating swap
                # (one that may break coverage) to escape local minimum.
                # Pick a random non-code word as a "perturbation" swap.
                if random.random() < T:
                    c_prime = random.randint(0, total - 1)
                    while c_prime in current_code:
                        c_prime = random.randint(0, total - 1)
                    do_swap = True
                else:
                    no_improve_counter += 1
                    continue
            else:
                c_prime = random.choice(valid_swaps)
                do_swap = True

            if do_swap:
                # Perform swap
                current_code.remove(c)
                current_code.add(c_prime)
                cnt[ball_indices_np[c]] -= 1
                cnt[ball_indices_np[c_prime]] += 1

                # If coverage broken, repair
                if cnt.min() < 1:
                    repair(current_code, cnt)

                # After swap, run delete phase
                removed = run_delete_phase(current_code, cnt)
                if removed > 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1

                # Cool down
                T = max(T_min, T * T_decay)

        # 3. Perturbation escape: if stuck, add a random codeword (inspiration)
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            rand_cw = random.randint(0, total - 1)
            if rand_cw not in current_code:
                current_code.add(rand_cw)
                cnt[ball_indices_np[rand_cw]] += 1
                # then try to delete to get back down
                run_delete_phase(current_code, cnt)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                # Reset temperature to re-explore
                T = 1.0
                no_improve_counter = 0

        # Occasionally try a fresh greedy restart if long stuck
        if no_improve_counter > max_no_improve * 3 and time.time() < deadline - 20.0:
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 3:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                T = 1.0

        if best_size <= 160:
            break

    # Final cleanup: remove any redundancy from best_code
    best_set = set(best_code)
    best_cnt = np.zeros(total, dtype=np.int32)
    for c in best_set:
        best_cnt[ball_indices_np[c]] += 1
    run_delete_phase(best_set, best_cnt)
    if best_cnt.min() >= 1:
        best_code = list(best_set)

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
