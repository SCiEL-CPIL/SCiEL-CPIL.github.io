"""Rewritten program: simulated annealing with vectorized delta for 4-AP minimization."""
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
    # (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N, d >= 1
    aps = []
    for d in range(1, N // 3 + 1):
        max_start = N - 3 * d
        for a in range(max_start):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # For each position, precompute which APs it participates in
    # pos_aps[i] = list of AP indices where position i appears
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(p, dtype=np.int32) if p else np.array([], dtype=np.int32) for p in pos_aps]
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is a numpy array of 0s and 1s."""
        c0 = coloring[aps_arr[:, 0]]
        c1 = coloring[aps_arr[:, 1]]
        c2 = coloring[aps_arr[:, 2]]
        c3 = coloring[aps_arr[:, 3]]
        # Monochromatic if all four are equal
        # (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def calc_delta(coloring, pos, new_val):
        """Calculate change in mono count if we flip position pos to new_val."""
        # Only APs containing pos can change
        if not pos_aps_arr[pos].size:
            return 0
        ap_indices = pos_aps_arr[pos]
        # Get the four colors for each of these APs
        a = coloring[aps_arr[ap_indices, 0]]
        b = coloring[aps_arr[ap_indices, 1]]
        c = coloring[aps_arr[ap_indices, 2]]
        d = coloring[aps_arr[ap_indices, 3]]
        
        # Current monochromatic status
        curr_mono = (a == b) & (b == c) & (c == d)
        
        # New colors: replace the color at pos with new_val
        # We need to know which position in the AP is pos
        # Let's just recompute by checking each AP
        # For efficiency, let's determine the position index
        a_idx = aps_arr[ap_indices, 0]
        b_idx = aps_arr[ap_indices, 1]
        c_idx = aps_arr[ap_indices, 2]
        d_idx = aps_arr[ap_indices, 3]
        
        # Mask for which slot pos is in
        is_a = (a_idx == pos)
        is_b = (b_idx == pos)
        is_c = (c_idx == pos)
        is_d = (d_idx == pos)
        
        new_a = np.where(is_a, new_val, a)
        new_b = np.where(is_b, new_val, b)
        new_c = np.where(is_c, new_val, c)
        new_d = np.where(is_d, new_val, d)
        
        new_mono = (new_a == new_b) & (new_b == new_c) & (new_c == new_d)
        
        return int(np.sum(new_mono) - np.sum(curr_mono))
    
    # Initialize with random coloring
    coloring = np.random.randint(0, 2, N).astype(np.int8)
    current_cost = count_mono(coloring)
    best_coloring = coloring.copy()
    best_cost = current_cost
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        # Try a random flip
        pos = random.randint(0, N - 1)
        new_val = 1 - int(coloring[pos])
        
        delta = calc_delta(coloring, pos, new_val)
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            coloring[pos] = new_val
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = coloring.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Greedy polish on best solution
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(N):
            new_val = 1 - int(best_coloring[pos])
            delta = calc_delta(best_coloring, pos, new_val)
            if delta < 0:
                best_coloring[pos] = new_val
                best_cost += delta
                improved = True
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
