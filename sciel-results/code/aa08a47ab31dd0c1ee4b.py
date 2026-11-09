# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0

    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)

    # data[d][r] = (pos_array, s_array, prefix_array, max_abs)
    data = {}
    global_max = 0

    for d in range(1, n + 1):
        data[d] = {}
        for r in range(d):
            if r >= n:
                continue
            pos = np.array(range(r, n, d), dtype=np.int32)
            m = len(pos)
            if m == 0:
                continue
            s_arr = signs[pos].copy()
            prefix = np.zeros(m + 1, dtype=np.int32)
            np.cumsum(s_arr, out=prefix[1:])
            # Kadane for max and min subarray
            max_e = max_f = 0
            min_e = min_f = 0
            for v in s_arr:
                max_e = max(0, max_e + v)
                max_f = max(max_f, max_e)
                min_e = min(0, min_e + v)
                min_f = min(min_f, min_e)
            max_abs = max(max_f, -min_f)
            data[d][r] = (pos, s_arr, prefix, max_abs)
            global_max = max(global_max, max_abs)

    best_signs = signs.copy()
    best_gm = global_max

    T = 2.0
    T_min = 0.05
    cooling = 0.9995

    while time.time() < deadline:
        i = rng.randrange(n)
        old_sign = signs[i]
        new_sign = -old_sign
        delta = 2 * old_sign  # sum changes by delta for each AP containing i

        # Apply flip
        signs[i] = new_sign
        new_gm = 0
        affected = []  # (d, r) pairs that were updated

        for d in range(1, n + 1):
            r = i % d
            if r not in data[d]:
                continue
            pos, s_arr, prefix, old_ma = data[d][r]
            k = i // d
            if k >= len(s_arr):
                continue
            s_arr[k] = new_sign
            prefix[k + 1:] += delta
            # Recompute max_abs
            max_e = max_f = 0
            min_e = min_f = 0
            for v in s_arr:
                max_e = max(0, max_e + v)
                max_f = max(max_f, max_e)
                min_e = min(0, min_e + v)
                min_f = min(min_f, min_e)
            new_ma = max(max_f, -min_f)
            data[d][r] = (pos, s_arr, prefix, new_ma)
            affected.append((d, r, old_ma))
            new_gm = max(new_gm, new_ma)

        # Accept or reject
        accept = (new_gm <= best_gm) or (rng.random() < np.exp(-(new_gm - best_gm) / T))
        if accept:
            best_gm = new_gm
            best_signs = signs.copy()
        else:
            # Revert
            signs[i] = old_sign
            for (d, r, old_ma) in affected:
                pos, s_arr, prefix, _ = data[d][r]
                k = i // d
                s_arr[k] = old_sign
                prefix[k + 1:] -= delta
                # Recompute
                max_e = max_f = 0
                min_e = min_f = 0
                for v in s_arr:
                    max_e = max(0, max_e + v)
                    max_f = max(max_f, max_e)
                    min_e = min(0, min_e + v)
                    min_f = min(min_f, min_e)
                new_ma = max(max_f, -min_f)
                data[d][r] = (pos, s_arr, prefix, new_ma)

        T *= cooling
        if T < T_min:
            T = T_min

    return list(best_signs)
# EVOLVE-BLOCK-END
