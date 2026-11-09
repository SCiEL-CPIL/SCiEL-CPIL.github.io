import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_aps]

    # Precompute column index for each position in each AP
    pos_col = []
    for pos in range(n):
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) > 0:
            affected = aps_arr[ap_indices]
            col_arr = np.argmax(affected == pos, axis=1)
            pos_col.append(col_arr)
        else:
            pos_col.append(np.array([], dtype=np.int32))

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        ap_indices = pos_aps_arr[pos]
        K = len(ap_indices)
        if K == 0:
            return 0
        affected = aps_arr[ap_indices]
        colors = col[affected]
        col_idx = pos_col[pos]
        cur_mono = (colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2]) & (colors[:, 2] == colors[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        flipped = colors.copy()
        rows = np.arange(K)
        flipped[rows, col_idx] = 1 - flipped[rows, col_idx]
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        return new_mono_count - cur_mono_count

    def greedy_polish(col, max_iters=150):
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            improved = False
            iters += 1
            order = np.random.permutation(n)
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    def sa_search(col, t0=5.0, t1=0.01, cooling=0.997, max_iters=500):
        col = col.copy()
        cur_count = count_mono(col)
        best_col = col.copy()
        best_count = cur_count
        T = t0
        for _ in range(max_iters):
            if (time.time() - start_time) > deadline - 5.0:
                break
            T = max(T * cooling, t1)
            pos = np.random.randint(n)
            new_col = col.copy()
            new_col[pos] = 1 - new_col[pos]
            d = get_delta(pos, col)
            if d < 0 or (T > t1 and random.random() < math.exp(-d / T)):
                col = new_col
                cur_count += d
                if cur_count < best_count:
                    best_count = cur_count
                    best_col = col.copy()
        return best_col, best_count

    # Phase 1: Multiple diverse initializations
    best_c = None
    best_count = float('inf')

    # Thue-Morse
    c_tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_tm)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # Random starts
    for _ in range(8):
        c_rand = np.random.randint(0, 2, n).astype(np.int8)
        c_polished, count = greedy_polish(c_rand)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # Block patterns
    for block_size in [2, 4, 8, 16, 32, 64]:
        c_block = np.zeros(n, dtype=np.int8)
        for i in range(n):
            c_block[i] = (i // block_size) % 2
        c_polished, count = greedy_polish(c_block)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # Phase 2: Iterated Local Search with SA perturbation
    while (time.time() - start_time) < deadline - 10.0:
        c_current = best_c.copy()
        
        time_remaining = deadline - 10.0 - (time.time() - start_time)
        if time_remaining < 15.0:
            num_flips = random.randint(8, 15)
        else:
            num_flips = random.randint(4, 12)
        
        positions_to_flip = np.random.choice(n, num_flips, replace=False)
        for p in positions_to_flip:
            c_current[p] = 1 - c_current[p]
        
        c_polished, count = greedy_polish(c_current)
        
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()
            # Double polish on improvement
            c_polished2, count2 = greedy_polish(c_polished)
            if count2 < best_count:
                best_count = count2
                best_c = c_polished2.copy()
        
        # Occasionally use SA to escape
        if random.random() < 0.3 and (time.time() - start_time) < deadline - 15.0:
            c_sa, sa_count = sa_search(c_current, t0=3.0, t1=0.01, cooling=0.995, max_iters=200)
            c_polished, count = greedy_polish(c_sa)
            if count < best_count:
                best_count = count
                best_c = c_polished.copy()

    # Phase 3: Final polish
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c, max_iters=200)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    actual_count = count_mono(best_c)
    return best_c.tolist()
