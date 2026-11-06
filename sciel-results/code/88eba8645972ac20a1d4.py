import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])

    num_aps = len(ap_indices)

    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)

    # Store AP indices as numpy arrays
    ap_arrays = [np.array(idx, dtype=np.int32) for idx in ap_indices]
    # Precompute as a 2D padded array with lengths for vectorized sum
    max_len = max(len(a) for a in ap_arrays)
    ap_padded = np.zeros((num_aps, max_len), dtype=np.int32)
    ap_lens = np.zeros(num_aps, dtype=np.int32)
    for k, arr in enumerate(ap_arrays):
        ap_padded[k, :len(arr)] = arr
        ap_lens[k] = len(arr)

    def compute_ap_sums(signs):
        # Vectorized: sum signs at each AP's positions
        # Use advanced indexing: for each AP, sum signs[ap_padded[k, :len_k]]
        # Trick: create mask or use np.sum with where
        # Simpler: loop but with numpy sum per AP (faster than Python sum)
        ap_sums = np.empty(num_aps, dtype=np.int32)
        for k in range(num_aps):
            ap_sums[k] = np.sum(signs[ap_padded[k, :ap_lens[k]]])
        return ap_sums

    def build_hist(ap_sums):
        hist = np.zeros(n + 1, dtype=np.int32)
        np.add.at(hist, np.abs(ap_sums), 1)
        return hist

    def get_max_from_hist(hist):
        # Find largest k with hist[k] > 0
        nz = np.nonzero(hist)[0]
        if len(nz) == 0:
            return 0
        return int(nz[-1])

    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    def local_search(signs, max_iters, deadline, seed):
        signs = signs.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        best_signs = signs.copy()
        best_max = current_max

        temperature = 2.5
        cooling = 0.9997
        rng = random.Random(seed)

        # Precompute affected AP arrays as numpy arrays for fast update
        pos_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Targeted selection: 70% chance pick from positions in worst APs
            if rng.random() < 0.7 and current_max > 0:
                # Find APs with max abs sum
                worst_aps = np.where(np.abs(ap_sums) == current_max)[0]
                if len(worst_aps) > 0:
                    # Pick a random worst AP, then a random position in it
                    ap_id = worst_aps[rng.randint(0, len(worst_aps) - 1)]
                    idx = ap_padded[ap_id, :ap_lens[ap_id]]
                    i = idx[rng.randint(0, len(idx) - 1)]
                else:
                    i = rng.randint(0, n - 1)
            else:
                i = rng.randint(0, n - 1)

            old_sign = signs[i]
            affected = pos_aps_arr[i]
            delta_sign = -2 * old_sign  # change in sum for each affected AP

            # Flip
            signs[i] = -old_sign

            # Vectorized update
            old_abs = np.abs(ap_sums[affected])
            hist[old_abs] -= 1
            ap_sums[affected] += delta_sign
            new_abs = np.abs(ap_sums[affected])
            hist[new_abs] += 1

            new_max = get_max_from_hist(hist)
            delta = new_max - current_max

            if delta <= 0 or (temperature > 0.01 and rng.random() < np.exp(-delta / temperature)):
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                ap_sums[affected] -= delta_sign
                new_abs2 = np.abs(ap_sums[affected])
                hist[new_abs2] -= 1
                old_abs2 = np.abs(ap_sums[affected])
                hist[old_abs2] += 1

            # 2-flip move every 50 iterations to escape local minima
            if it % 50 == 49 and temperature > 0.1:
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 != i2:
                    old1, old2 = signs[i1], signs[i2]
                    aff1, aff2 = pos_aps_arr[i1], pos_aps_arr[i2]
                    signs[i1], signs[i2] = -old1, -old2

                    # Update both
                    all_aff = np.unique(np.concatenate([aff1, aff2]))
                    old_abs_all = np.abs(ap_sums[all_aff])
                    hist[old_abs_all] -= 1

                    # Compute new sums for affected APs
                    for ap_id in all_aff:
                        arr = ap_padded[ap_id, :ap_lens[ap_id]]
                        ap_sums[ap_id] = int(np.sum(signs[arr]))

                    new_abs_all = np.abs(ap_sums[all_aff])
                    hist[new_abs_all] += 1

                    new_max2 = get_max_from_hist(hist)
                    delta2 = new_max2 - current_max

                    if delta2 <= 0 or (temperature > 0.01 and rng.random() < np.exp(-delta2 / temperature)):
                        current_max = new_max2
                        if current_max < best_max:
                            best_max = current_max
                            best_signs = signs.copy()
                    else:
                        # Revert
                        signs[i1], signs[i2] = old1, old2
                        for ap_id in all_aff:
                            arr = ap_padded[ap_id, :ap_lens[ap_id]]
                            ap_sums[ap_id] = int(np.sum(signs[arr]))
                        new_abs_revert = np.abs(ap_sums[all_aff])
                        hist[new_abs_revert] -= 1
                        old_abs_revert = np.abs(ap_sums[all_aff])
                        hist[old_abs_revert] += 1

            temperature *= cooling

        return best_signs, best_max

    best_overall = None
    best_overall_max = n + 1

    initial_signs_list = []

    # Thue-Morse and variants
    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    initial_signs_list.append(-tm)

    # Shifted Thue-Morse
    for shift in [1, 2, 3]:
        shifted = np.roll(tm, shift)
        initial_signs_list.append(shifted)

    # Random balanced sequences
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999, 5555, 7777, 31337, 27182, 14142]
    for seed in seeds:
        rng = random.Random(seed)
        # Generate balanced: 128 +1 and 128 -1
        arr = [1]*128 + [-1]*128
        rng.shuffle(arr)
        initial_signs_list.append(np.array(arr, dtype=np.int8))

    for signs_init in initial_signs_list:
        if time.time() > deadline - 5.0:
            break
        seed = hash(signs_init.tobytes()) % (2**31)
        signs, mx = local_search(signs_init, max_iters=4000, deadline=deadline, seed=seed)
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break

    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        pos_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                affected = pos_aps_arr[i]
                delta_sign = -2 * old_sign

                signs[i] = -old_sign
                old_abs = np.abs(ap_sums[affected])
                hist[old_abs] -= 1
                ap_sums[affected] += delta_sign
                new_abs = np.abs(ap_sums[affected])
                hist[new_abs] += 1

                new_max = get_max_from_hist(hist)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                    ap_sums[affected] -= delta_sign
                    new_abs2 = np.abs(ap_sums[affected])
                    hist[new_abs2] -= 1
                    old_abs2 = np.abs(ap_sums[affected])
                    hist[old_abs2] += 1

        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()

    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]

    return [int(s) for s in best_overall]
