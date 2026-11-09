import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    M = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Build per-position list of AP indices and the indices of the OTHER elements in those APs
    pos_aps = [[] for _ in range(n)]
    pos_others = [[] for _ in range(n)]
    
    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            pos_aps[p].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_others[p].append(others)

    # Precompute for vectorized delta: for each position, store the AP indices and other positions
    # pos_ap_indices[p] = np.array of ap indices where p appears
    # pos_other_indices[p] = np.array of shape (len, 3) with the other 3 positions for each ap
    pos_ap_indices = [np.array(pos_aps[p], dtype=np.int32) for p in range(n)]
    pos_other_indices = [np.array(pos_others[p], dtype=np.int32) for p in range(n)]

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta_vec(pos, col):
        """Vectorized delta calculation for flipping col[pos]."""
        ap_idxs = pos_ap_indices[pos]
        if len(ap_idxs) == 0:
            return 0
        others = pos_other_indices[pos]  # shape (K, 3)
        c1 = col[others[:, 0]]
        c2 = col[others[:, 1]]
        c3 = col[others[:, 2]]
        # Check if the other 3 are all equal
        all_equal = (c1 == c2) & (c2 == c3)
        # If all equal: if col[pos] == c1, flipping removes mono (delta -= 1), else adds (delta += 1)
        col_pos = col[pos]
        delta_remove = all_equal & (col_pos == c1)
        delta_add = all_equal & (col_pos != c1)
        return int(np.sum(delta_add) - np.sum(delta_remove))

    def get_delta_swap_vec(i, j, col):
        """Vectorized delta for swapping (flipping) col[i] and col[j]."""
        # Get union of AP indices affected by i or j
        ai = pos_ap_indices[i]
        aj = pos_ap_indices[j]
        if len(ai) == 0 and len(aj) == 0:
            return 0
        if len(ai) == 0:
            affected = aj
        elif len(aj) == 0:
            affected = ai
        else:
            affected = np.unique(np.concatenate([ai, aj]))
        
        if len(affected) == 0:
            return 0
        
        # Get the 4 positions for each affected AP
        ap0 = aps_arr[affected, 0]
        ap1 = aps_arr[affected, 1]
        ap2 = aps_arr[affected, 2]
        ap3 = aps_arr[affected, 3]
        
        # Get current colors
        c0 = col[ap0].astype(np.int8)
        c1 = col[ap1].astype(np.int8)
        c2 = col[ap2].astype(np.int8)
        c3 = col[ap3].astype(np.int8)
        
        was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Apply flips: flip positions that are i or j
        flip0 = (ap0 == i) | (ap0 == j)
        flip1 = (ap1 == i) | (ap1 == j)
        flip2 = (ap2 == i) | (ap2 == j)
        flip3 = (ap3 == i) | (ap3 == j)
        
        nc0 = np.where(flip0, 1 - c0, c0)
        nc1 = np.where(flip1, 1 - c1, c1)
        nc2 = np.where(flip2, 1 - c2, c2)
        nc3 = np.where(flip3, 1 - c3, c3)
        
        is_mono = (nc0 == nc1) & (nc1 == nc2) & (nc2 == nc3)
        
        delta = int(np.sum(is_mono & ~was_mono) - np.sum(was_mono & ~is_mono))
        return delta

    def greedy_polish(col, use_swap=False, max_rounds=50):
        """Run greedy local search until no improvement. Optionally allow 2-flip swaps."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and (time.time() - start_time) < deadline - 5.0:
            improved = False
            rounds += 1
            order = list(range(n))
            random.shuffle(order)
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta_vec(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
            # If using swap, try random swaps
            if use_swap:
                for _ in range(20):
                    if (time.time() - start_time) > deadline - 5.0:
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j: continue
                    d = get_delta_swap_vec(i, j, col)
                    if d < 0:
                        col[i] = 1 - col[i]
                        col[j] = 1 - col[j]
                        cur_count += d
                        improved = True
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, best_count = greedy_polish(c, use_swap=True, max_rounds=20)
    best_c = c.copy()

    # Phase 2: Simulated Annealing with vectorized deltas
    T = 3.0
    cooling = 0.99995
    min_T = 0.02
    no_improve_steps = 0
    max_no_improve = 5000
    
    while (time.time() - start_time) < deadline - 3.0:
        # Pick random position
        pos = random.randint(0, n - 1)
        d = get_delta_vec(pos, c)

        accept = False
        if d < 0:
            accept = True
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-d / T)
                if random.random() < prob:
                    accept = True
                    no_improve_steps = 0

        if accept:
            c[pos] = 1 - c[pos]
            best_count += d
            # Occasionally check if we improved the best
            if no_improve_steps % 100 == 0 or best_count < count_mono(best_c):
                if best_count < count_mono(best_c):
                    best_c = c.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
        else:
            no_improve_steps += 1

        T *= cooling

        # Restart strategy if stuck
        if no_improve_steps > max_no_improve:
            # Reset to best
            c = best_c.copy()
            best_count = count_mono(c)
            T = 2.0
            no_improve_steps = 0
            # Perturb: flip random positions
            for _ in range(3 + random.randint(0, 4)):
                p2 = random.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            best_count = count_mono(c)
            # Polish to get back on track
            c, best_count = greedy_polish(c, use_swap=True, max_rounds=10)
            if best_count < count_mono(best_c):
                best_c = c.copy()

    # Phase 3: Final high-precision polish - multiple passes
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c, use_swap=True, max_rounds=100)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count
        
        # One more round of SA with very low temperature for fine-tuning
        if (time.time() - start_time) < deadline - 1.0:
            c = best_c.copy()
            best_count = count_mono(c)
            T = 0.05
            cooling = 0.9999
            min_T = 0.01
            steps = 0
            while (time.time() - start_time) < deadline - 0.5 and steps < 5000:
                pos = random.randint(0, n - 1)
                d = get_delta_vec(pos, c)
                if d < 0:
                    c[pos] = 1 - c[pos]
                    best_count += d
                    if best_count < count_mono(best_c):
                        best_c = c.copy()
                elif d == 0 and random.random() < 0.5:
                    c[pos] = 1 - c[pos]
                    best_count += d
                elif T > min_T and random.random() < math.exp(-d / T):
                    c[pos] = 1 - c[pos]
                    best_count += d
                T *= cooling
                steps += 1

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
