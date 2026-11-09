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

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Calculate delta in mono count if we flip col[pos]."""
        delta = 0
        others_list = pos_others[pos]
        for others in others_list:
            c1 = col[others[0]]
            c2 = col[others[1]]
            c3 = col[others[2]]
            if c1 == c2 and c2 == c3:
                if col[pos] == c1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_delta_swap(i, j, col):
        """Calculate delta in mono count if we flip col[i] and col[j]."""
        delta = 0
        # Get all APs affected by i or j
        affected_aps = set(pos_aps[i])
        affected_aps.update(pos_aps[j])
        
        for ap_idx in affected_aps:
            ap = aps_arr[ap_idx]
            c0 = col[ap[0]]
            c1 = col[ap[1]]
            c2 = col[ap[2]]
            c3 = col[ap[3]]
            was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            # Apply flips
            if ap[0] == i or ap[0] == j: c0 = 1 - c0
            if ap[1] == i or ap[1] == j: c1 = 1 - c1
            if ap[2] == i or ap[2] == j: c2 = 1 - c2
            if ap[3] == i or ap[3] == j: c3 = 1 - c3
            
            is_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            if was_mono and not is_mono: delta -= 1
            elif not was_mono and is_mono: delta += 1
            
        return delta

    def greedy_polish(col, max_passes=50):
        """Run greedy local search (1-opt and 2-opt) until no improvement or max_passes."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        passes = 0
        while improved and passes < max_passes and (time.time() - start_time) < deadline - 5.0:
            improved = False
            passes += 1
            
            # 1-opt pass
            for pos in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
            
            # 2-opt pass (only a subset for speed)
            for i in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    d = get_delta_swap(i, j, col)
                    if d < 0:
                        col[i] = 1 - col[i]
                        col[j] = 1 - col[j]
                        cur_count += d
                        improved = True
                        
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, best_count = greedy_polish(c)
    best_c = c.copy()
    best_count = count_mono(best_c)

    # Phase 2: Simulated Annealing with better parameters
    T = 3.0
    cooling = 0.9998
    min_T = 0.02
    no_improve_steps = 0
    max_no_improve = 3000
    
    # Precompute random for speed
    rng = random.Random(42)
    
    while (time.time() - start_time) < deadline - 3.0:
        # Pick random move: 1-bit flip or 2-bit swap
        use_swap = rng.random() < 0.5
        d = 0
        if use_swap:
            i, j = rng.sample(range(n), 2)
            d = get_delta_swap(i, j, c)
        else:
            i = rng.randint(0, n - 1)
            j = -1
            d = get_delta(i, c)

        accept = False
        if d < 0:
            accept = True
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-d / T)
                if rng.random() < prob:
                    accept = True
                    no_improve_steps = 0

        if accept:
            if use_swap:
                c[i] = 1 - c[i]
                c[j] = 1 - c[j]
            else:
                c[i] = 1 - c[i]
            
            best_count += d
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
            for _ in range(8):
                p2 = rng.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            best_count = count_mono(c)
            # Polish a bit to get back on track
            c, best_count = greedy_polish(c, max_passes=20)
            if best_count < count_mono(best_c):
                best_c = c.copy()

    # Phase 3: Final greedy polish
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c, max_passes=100)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
