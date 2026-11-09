import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 256

    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    aps_arr = np.array(aps, dtype=np.int32)
    
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)
        
    pair_aps = {}
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        inds = [i0, i1, i2, i3]
        for a in range(4):
            for b in range(a+1, 4):
                p, q = inds[a], inds[b]
                key = (p, q) if p < q else (q, p)
                if key not in pair_aps:
                    pair_aps[key] = []
                pair_aps[key].append(idx)

    def count_mono(color):
        cols = color[aps_arr]
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    def flip_delta(color, p):
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = color[p]
        c_new = 1 - c_p
        others_colors = color[others]
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))

    def flip_pair_delta(color, p, q):
        delta_p = flip_delta(color, p)
        delta_q = flip_delta(color, q)
        
        key = (p, q) if p < q else (q, p)
        common_aps = pair_aps.get(key, [])
        
        if not common_aps:
            return delta_p + delta_q
            
        c_p = color[p]
        c_q = color[q]
        new_c_p = 1 - c_p
        new_c_q = 1 - c_q
        
        interaction_sum = 0
        
        for ap_idx in common_aps:
            i0, i1, i2, i3 = aps_arr[ap_idx]
            
            others = []
            for idx in (i0, i1, i2, i3):
                if idx != p and idx != q:
                    others.append(idx)
            
            if len(others) != 2:
                continue # Should not happen for valid APs containing p and q
                
            u, v = others
            c_u = color[u]
            c_v = color[v]
            
            # Check mono status
            # Old: c_p, c_q, c_u, c_v
            # P:    new_c_p, c_q, c_u, c_v
            # Q:    c_p, new_c_q, c_u, c_v
            # PQ:   new_c_p, new_c_q, c_u, c_v
            
            # Mono if all 4 equal
            
            # Optimization:
            # If c_u != c_v, none of the states can be mono (since u and v are fixed and different).
            if c_u != c_v:
                continue
                
            # If c_u == c_v, let's call this color C.
            # Mono if cp == C and cq == C.
            
            c_fixed = c_u
            
            cost_old = 1 if (c_p == c_fixed and c_q == c_fixed) else 0
            cost_p = 1 if (new_c_p == c_fixed and c_q == c_fixed) else 0
            cost_q = 1 if (c_p == c_fixed and new_c_q == c_fixed) else 0
            cost_pq = 1 if (new_c_p == c_fixed and new_c_q == c_fixed) else 0
            
            interaction_sum += cost_pq - cost_p - cost_q + cost_old
            
        return delta_p + delta_q + interaction_sum
    
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color

    def simulated_annealing(color):
        color = color.copy()
        current_score = count_mono(color)
        best_color = color.copy()
        best_score = current_score
        
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.999
        
        T = T_start
        
        while time.time() - start_time < deadline - 2 and T > T_end:
            if np.random.rand() < 0.5:
                p = np.random.randint(n)
                q = np.random.randint(n)
                while q == p:
                    q = np.random.randint(n)
                delta = flip_pair_delta(color, p, q)
                
                if delta < 0:
                    accept = True
                else:
                    accept = (np.random.rand() < np.exp(-delta / T))
                
                if accept:
                    color[p] = 1 - color[p]
                    color[q] = 1 - color[q]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
            else:
                p = np.random.randint(n)
                delta = flip_delta(color, p)
                
                if delta < 0:
                    accept = True
                else:
                    accept = (np.random.rand() < np.exp(-delta / T))
                
                if accept:
                    color[p] = 1 - color[p]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color.copy()
            
            T *= cooling_rate
            
        return best_color, best_score

    startings = [thue_morse(n)]
    for bs in [2, 4, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(5):
        startings.append(random_coloring())
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=15, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)

    scored_startings = []
    for sc in startings:
        sc_score = count_mono(sc)
        scored_startings.append((sc_score, sc))
    scored_startings.sort(key=lambda x: x[0])
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    runs_per_start = 50
    
    for sc_score, sc in scored_startings:
        for _ in range(runs_per_start):
            if time.time() - start_time > deadline - 5:
                break
            color, score = simulated_annealing(sc)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
        
        if time.time() - start_time > deadline - 5:
            break

    if time.time() - start_time < deadline - 10 and best_overall_color is not None:
        while time.time() - start_time < deadline - 2:
            color, score = simulated_annealing(best_overall_color)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()

    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
