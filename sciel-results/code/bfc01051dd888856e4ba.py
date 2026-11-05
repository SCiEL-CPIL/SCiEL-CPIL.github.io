"""Combined: vectorized-delta SA + multi-restart + pair-flip polish for 4-AP minimization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    random.seed(12345)
    np.random.seed(12345)
    
    N = 256
    
    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, N // 3 + 1):
        max_start = N - 3 * d
        for a in range(max_start):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # For each position, precompute which APs it participates in and its role
    pos_aps = [[] for _ in range(N)]  # list of (ap_idx, role)
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    
    # Convert to arrays: for each pos, store ap_indices and roles
    pos_ap_idx = [np.array([x[0] for x in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_aps]
    pos_ap_role = [np.array([x[1] for x in lst], dtype=np.int8) if lst else np.array([], dtype=np.int8) for lst in pos_aps]
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is a numpy array of 0s and 1s."""
        c0 = coloring[aps_arr[:, 0]]
        c1 = coloring[aps_arr[:, 1]]
        c2 = coloring[aps_arr[:, 2]]
        c3 = coloring[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def calc_delta(coloring, pos, new_val):
        """Calculate change in mono count if we flip position pos to new_val."""
        ap_indices = pos_ap_idx[pos]
        if ap_indices.size == 0:
            return 0
        roles = pos_ap_role[pos]
        
        a = coloring[aps_arr[ap_indices, 0]]
        b = coloring[aps_arr[ap_indices, 1]]
        c = coloring[aps_arr[ap_indices, 2]]
        d = coloring[aps_arr[ap_indices, 3]]
        
        curr_mono = (a == b) & (b == c) & (c == d)
        
        # Build new colors by replacing the slot at pos
        new_a = np.where(roles == 0, new_val, a)
        new_b = np.where(roles == 1, new_val, b)
        new_c = np.where(roles == 2, new_val, c)
        new_d = np.where(roles == 3, new_val, d)
        
        new_mono = (new_a == new_b) & (new_b == new_c) & (new_c == new_d)
        
        return int(np.sum(new_mono) - np.sum(curr_mono))
    
    def greedy_polish(coloring, max_passes=50):
        """Best-improvement 1-flip local search."""
        coloring = coloring.copy()
        current_cost = count_mono(coloring)
        
        for _ in range(max_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            best_delta = 0
            best_pos = -1
            for pos in range(N):
                new_val = 1 - int(coloring[pos])
                delta = calc_delta(coloring, pos, new_val)
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            if best_pos >= 0:
                coloring[best_pos] = 1 - int(coloring[best_pos])
                current_cost += best_delta
                improved = True
            if not improved:
                break
        
        return coloring, current_cost
    
    def pair_flip_polish(coloring, max_passes=30):
        """Try flipping pairs of positions to escape 1-flip local optima."""
        coloring = coloring.copy()
        current_cost = count_mono(coloring)
        
        for _ in range(max_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            # Sample a subset of pairs to keep it fast
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                p1, p2 = random.sample(range(N), 2)
                old1, old2 = coloring[p1], coloring[p2]
                coloring[p1] = 1 - old1
                coloring[p2] = 1 - old2
                new_cost = count_mono(coloring)
                delta = new_cost - current_cost
                if delta < 0:
                    current_cost = new_cost
                    improved = True
                else:
                    coloring[p1] = old1
                    coloring[p2] = old2
            if not improved:
                break
        
        return coloring, current_cost
    
    def simulated_annealing(initial_colors, start_temp=1.0, end_temp=0.005,
                            cooling_rate=0.9995, max_steps=50000):
        """Simulated annealing with vectorized delta."""
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            pos = random.randint(0, N - 1)
            new_val = 1 - int(colors[pos])
            
            delta = calc_delta(colors, pos, new_val)
            
            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                colors[pos] = new_val
                current += delta
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count
    
    # Strategy: Multiple restarts with diverse initial colorings
    best_colors = None
    best_score = float('inf')
    
    # Initial colorings to try
    initial_colorings = []
    
    # 1. Random
    initial_colorings.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # 2. Balanced alternating
    alt = np.zeros(N, dtype=np.int8)
    alt[::2] = 0
    alt[1::2] = 1
    initial_colorings.append(alt.copy())
    
    # 3. Block-based: 0 for 0-127, 1 for 128-255
    block = np.zeros(N, dtype=np.int8)
    block[128:] = 1
    initial_colorings.append(block.copy())
    
    # 4. Another random
    initial_colorings.append(np.random.randint(0, 2, N, dtype=np.int8))
    
    # 5. Pattern: 0,0,1,1 repeating
    pat = np.zeros(N, dtype=np.int8)
    for i in range(N):
        pat[i] = (i // 2) % 2
    initial_colorings.append(pat.copy())
    
    for init_colors in initial_colorings:
        if time.time() > deadline - 15:
            break
        
        # Greedy polish first
        polished, score = greedy_polish(init_colors, max_passes=30)
        
        if score < best_score:
            best_score = score
            best_colors = polished.copy()
        
        # Simulated annealing from polished state
        if time.time() < deadline - 10:
            sa_colors, sa_score = simulated_annealing(
                polished, start_temp=0.5, end_temp=0.005,
                cooling_rate=0.9995, max_steps=30000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
            
            # Pair-flip polish after SA
            if time.time() < deadline - 5:
                pf_colors, pf_score = pair_flip_polish(sa_colors, max_passes=20)
                if pf_score < best_score:
                    best_score = pf_score
                    best_colors = pf_colors.copy()
    
    # Final polish on the best solution
    if time.time() < deadline - 3:
        final_colors, final_score = greedy_polish(best_colors, max_passes=50)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    # Pair-flip on final best
    if time.time() < deadline - 2:
        pf_colors, pf_score = pair_flip_polish(best_colors, max_passes=20)
        if pf_score < best_score:
            best_score = pf_score
            best_colors = pf_colors.copy()
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
