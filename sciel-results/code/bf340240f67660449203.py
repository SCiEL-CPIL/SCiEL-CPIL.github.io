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

    # For vectorized delta: for each position, store the AP indices it belongs to
    # and for each such AP, the other 3 positions
    # Build arrays: for position p, we need to know for each AP containing p,
    # whether the other 3 are all equal, and if so, whether p matches them.
    
    # Precompute for each position: list of (ap_idx, other1, other2, other3)
    pos_ap_info = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            others = [ap[j] for j in range(4) if j != k]
            pos_ap_info[p].append((idx, others[0], others[1], others[2]))

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_deltas_vectorized(col):
        """Calculate delta for ALL positions at once. Returns array of length n."""
        # For each position p, delta = sum over APs containing p of:
        #   if other 3 are all equal: -1 if col[p]==col[other], +1 otherwise
        #   else: 0
        # 
        # We can compute this by iterating over APs and accumulating.
        # For each AP (i0,i1,i2,i3), if all 4 are same color, flipping any one breaks it (-1 for each of 4 positions)
        # If exactly 3 are same and 1 differs, flipping the odd one makes it mono (+1 for that position),
        #   flipping any of the 3 majority makes it non-mono (-1 for each of 3)
        # Otherwise (2-2 split), flipping any position keeps it non-mono (0)
        
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        
        deltas = np.zeros(n, dtype=np.int32)
        
        # Case 1: all 4 same -> each position gets -1
        all_same = (a0 == a1) & (a1 == a2) & (a2 == a3)
        if np.any(all_same):
            for idx in np.where(all_same)[0]:
                ap = aps_arr[idx]
                deltas[ap[0]] -= 1
                deltas[ap[1]] -= 1
                deltas[ap[2]] -= 1
                deltas[ap[3]] -= 1
        
        # Case 2: exactly 3 same, 1 different
        # The odd one out gets +1, the three majority get -1 each
        # Determine which position is the odd one
        # pos0 is odd if a0 != a1 and a1==a2 and a2==a3
        # pos1 is odd if a1 != a0 and a0==a2 and a2==a3 (but need to be careful)
        # Better: for each AP, find the minority element
        
        # Compute for each AP: how many are 0 and how many are 1
        s = a0 + a1 + a2 + a3  # sum of colors, 0 to 4
        
        # 3 zeros, 1 one: the one is the odd one (gets +1), zeros get -1
        # 3 ones, 1 zero: the zero is the odd one (gets +1), ones get -1
        # 2-2 split: no contribution
        
        mask_31 = (s == 3)  # three 1s, one 0 -> the 0 is odd
        mask_13 = (s == 1)  # three 0s, one 1 -> the 1 is odd
        
        if np.any(mask_31):
            for idx in np.where(mask_31)[0]:
                ap = aps_arr[idx]
                # Find which position has color 0
                vals = [a0[idx], a1[idx], a2[idx], a3[idx]]
                for k, v in enumerate(vals):
                    if v == 0:
                        deltas[ap[k]] += 1  # odd one gets +1
                        break
                for k, v in enumerate(vals):
                    if v == 1:
                        deltas[ap[k]] -= 1
        
        if np.any(mask_13):
            for idx in np.where(mask_13)[0]:
                ap = aps_arr[idx]
                vals = [a0[idx], a1[idx], a2[idx], a3[idx]]
                for k, v in enumerate(vals):
                    if v == 1:
                        deltas[ap[k]] += 1
                        break
                for k, v in enumerate(vals):
                    if v == 0:
                        deltas[ap[k]] -= 1
                        
        return deltas

    def get_delta(pos, col):
        """Calculate delta in mono count if we flip col[pos]."""
        delta = 0
        for ap_idx, o1, o2, o3 in pos_ap_info[pos]:
            c1 = col[o1]
            c2 = col[o2]
            c3 = col[o3]
            if c1 == c2 and c2 == c3:
                if col[pos] == c1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_delta_swap(i, j, col):
        """Calculate delta in mono count if we flip col[i] and col[j]."""
        delta = 0
        affected_aps = set(pos_aps[i])
        affected_aps.update(pos_aps[j])
        
        for ap_idx in affected_aps:
            ap = aps_arr[ap_idx]
            c0 = col[ap[0]]
            c1 = col[ap[1]]
            c2 = col[ap[2]]
            c3 = col[ap[3]]
            was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            if ap[0] == i or ap[0] == j: c0 = 1 - c0
            if ap[1] == i or ap[1] == j: c1 = 1 - c1
            if ap[2] == i or ap[2] == j: c2 = 1 - c2
            if ap[3] == i or ap[3] == j: c3 = 1 - c3
            
            is_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    def greedy_polish_vectorized(col, use_swap=False, time_budget=5.0):
        """Run greedy local search using vectorized deltas."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        end_time = time.time() + time_budget
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            deltas = get_deltas_vectorized(col)
            # Find all positions with negative delta
            improving = np.where(deltas < 0)[0]
            if len(improving) > 0:
                random.shuffle(improving.tolist())
                for pos in improving:
                    if (time.time() - start_time) > deadline - 5.0:
                        break
                    col[pos] = 1 - col[pos]
                    cur_count += deltas[pos]
                    improved = True
            if use_swap:
                for _ in range(20):
                    if (time.time() - start_time) > deadline - 5.0:
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j: continue
                    d = get_delta_swap(i, j, col)
                    if d < 0:
                        col[i] = 1 - col[i]
                        col[j] = 1 - col[j]
                        cur_count += d
                        improved = True
        return col, cur_count

    def sa_search(col, T_init=2.0, cooling=0.99995, min_T=0.03, max_no_improve=3000, time_budget=20.0):
        """Simulated annealing with adaptive restarts."""
        col = col.copy()
        cur_count = count_mono(col)
        best_col = col.copy()
        best_count = cur_count
        T = T_init
        no_improve_steps = 0
        end_time = time.time() + time_budget
        
        while (time.time() - start_time) < deadline - 3.0 and (time.time() < end_time):
            pos = random.randint(0, n - 1)
            d = get_delta(pos, col)

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
                col[pos] = 1 - col[pos]
                cur_count += d
                if cur_count < best_count:
                    best_col = col.copy()
                    best_count = cur_count
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

            T *= cooling

            if no_improve_steps > max_no_improve:
                col = best_col.copy()
                cur_count = best_count
                T = max(T_init * 0.5, 0.5)
                no_improve_steps = 0
                # Larger perturbation
                n_flips = random.randint(5, 12)
                for _ in range(n_flips):
                    p2 = random.randint(0, n - 1)
                    col[p2] = 1 - col[p2]
                cur_count = count_mono(col)
                # Quick polish
                col, cur_count = greedy_polish_vectorized(col, use_swap=True, time_budget=2.0)
                if cur_count < best_count:
                    best_col = col.copy()
                    best_count = cur_count

        return best_col, best_count

    # Generate diverse initial colorings
    def thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def random_col():
        return np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int8)
    
    def structured_col(period):
        """Coloring with periodic structure."""
        col = np.zeros(n, dtype=np.int8)
        for i in range(n):
            col[i] = (i // period) % 2
        return col

    # Multi-restart strategy
    best_c = None
    best_count = float('inf')
    
    # Initial diverse starts
    initializations = [
        thue_morse(),
        random_col(),
        structured_col(4),
        structured_col(8),
        structured_col(16),
        random_col(),
    ]
    
    # Phase 1: Polish each initialization
    for init in initializations:
        if (time.time() - start_time) > deadline - 30.0:
            break
        polished, cnt = greedy_polish_vectorized(init, use_swap=True, time_budget=3.0)
        if cnt < best_count:
            best_count = cnt
            best_c = polished.copy()
    
    # Phase 2: SA from best and some other good starts
    if best_c is not None:
        sa_c, sa_cnt = sa_search(best_c, T_init=2.0, time_budget=15.0)
        if sa_cnt < best_count:
            best_count = sa_cnt
            best_c = sa_c.copy()
    
    # Try SA from a few other initializations if time permits
    if (time.time() - start_time) < deadline - 25.0:
        for init in initializations[1:4]:
            if (time.time() - start_time) > deadline - 20.0:
                break
            polished, cnt = greedy_polish_vectorized(init, use_swap=True, time_budget=2.0)
            if cnt < best_count:
                best_count = cnt
                best_c = polished.copy()
            sa_c, sa_cnt = sa_search(polished, T_init=1.5, time_budget=8.0)
            if sa_cnt < best_count:
                best_count = sa_cnt
                best_c = sa_c.copy()

    # Phase 3: More SA iterations from best
    if (time.time() - start_time) < deadline - 10.0 and best_c is not None:
        sa_c, sa_cnt = sa_search(best_c, T_init=1.0, cooling=0.99999, min_T=0.01, max_no_improve=5000, time_budget=15.0)
        if sa_cnt < best_count:
            best_count = sa_cnt
            best_c = sa_c.copy()

    # Phase 4: Final polish
    if (time.time() - start_time) < deadline - 3.0 and best_c is not None:
        final_c, final_count = greedy_polish_vectorized(best_c, use_swap=True, time_budget=5.0)
        if final_count < best_count:
            best_count = final_count
            best_c = final_c.copy()

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
