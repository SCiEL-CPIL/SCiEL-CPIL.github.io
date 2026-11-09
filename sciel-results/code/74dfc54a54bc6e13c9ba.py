import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin
    
    n = 256
    
    # Precompute all 4-term APs
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
    
    # Precompute neighbors for each position for fast delta
    # pos_neighbors[i] is a 2D array of shape (num_aps_containing_i, 3)
    # Each row contains the indices of the other 3 positions in that AP
    pos_neighbors = [None] * n
    for i in range(n):
        # Find all APs containing i
        # We can do this by checking which column of aps_arr has i
        # But it's faster to build it in one pass over aps
        pass
        
    # One pass to build pos_neighbors
    # First, count how many APs contain each position
    ap_counts = np.zeros(n, dtype=np.int32)
    for ap in aps_arr:
        for idx in ap:
            ap_counts[idx] += 1
            
    # Allocate arrays
    pos_neighbors = [np.empty((ap_counts[i], 3), dtype=np.int32) for i in range(n)]
    # Keep track of fill index
    fill_idx = np.zeros(n, dtype=np.int32)
    
    for ap in aps_arr:
        i0, i1, i2, i3 = ap
        # For i0, neighbors are i1, i2, i3
        idx = fill_idx[i0]
        pos_neighbors[i0][idx] = [i1, i2, i3]
        fill_idx[i0] += 1
        
        # For i1, neighbors are i0, i2, i3
        idx = fill_idx[i1]
        pos_neighbors[i1][idx] = [i0, i2, i3]
        fill_idx[i1] += 1
        
        # For i2, neighbors are i0, i1, i3
        idx = fill_idx[i2]
        pos_neighbors[i2][idx] = [i0, i1, i3]
        fill_idx[i2] += 1
        
        # For i3, neighbors are i0, i1, i2
        idx = fill_idx[i3]
        pos_neighbors[i3][idx] = [i0, i1, i2]
        fill_idx[i3] += 1
        
    # Initial coloring: Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Vectorized count for initial eval
    def count_mono_vec(c):
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))
        
    current_score = count_mono_vec(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # SA Parameters
    # We need to estimate the scale of delta.
    # Max delta per flip is roughly the number of APs containing i (max ~30-40).
    # T should start high enough to allow large jumps.
    T = 10.0
    T_min = 0.05
    # We want to run as many steps as possible.
    # Let's estimate steps. If delta calc is ~1-5us, we can do 100k-1M steps in 10s.
    # With 110s budget, we can do a lot.
    
    # Multi-start strategy:
    # Run SA for a fixed time chunk, then restart if no improvement for a while?
    # Or just one long run?
    # Let's do a few restarts to escape local minima.
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        # If first restart, use Thue-Morse. Else, use random or best so far?
        # Using best so far for restart might get stuck. Random is better for diversity.
        # But Thue-Morse is a very good start.
        if r == 0:
            colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
            current_score = count_mono_vec(colors)
        else:
            # Random restart, but maybe biased?
            # Let's try random restarts
            colors = np.random.randint(0, 2, n, dtype=np.int8)
            current_score = count_mono_vec(colors)
            
            # If random is much worse, maybe we should start from best?
            # No, SA from random can find different basins.
            
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9999 # Slower cooling to stay hot longer?
        
        # Actually, let's tune cooling.
        # If we want ~100k steps, and T goes from 5 to 0.01:
        # 0.01 = 5 * (cooling_rate)^100000
        # cooling_rate = (0.002)^(1/100000) ~ 0.999954
        
        # Let's just run until time limit for this restart
        restart_deadline = time.time() + time_per_restart
        last_improve_time = time.time()
        
        while time.time() < restart_deadline:
            if T < T_min:
                break
                
            # Pick random position
            i = np.random.randint(n)
            
            # Calculate delta
            old_val = colors[i]
            new_val = 1 - old_val
            
            neighbors = pos_neighbors[i]
            if len(neighbors) == 0:
                delta = 0
            else:
                c_n = colors[neighbors]
                # c_n shape (K, 3)
                all_same = (c_n[:, 0] == c_n[:, 1]) & (c_n[:, 1] == c_n[:, 2])
                # old_mono: all_same AND old_val == c_n[:,0]
                old_mono = all_same & (old_val == c_n[:, 0])
                # new_mono: all_same AND new_val == c_n[:,0]
                new_mono = all_same & (new_val == c_n[:, 0])
                
                delta = np.sum(new_mono) - np.sum(old_mono)
            
            # Acceptance
            if delta < 0:
                colors[i] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    last_improve_time = time.time()
            else:
                if delta == 0:
                    if np.random.random() < 0.5:
                        colors[i] = new_val
                else:
                    # exp(-delta / T)
                    # If delta is large and T small, this is 0.
                    # Optimize: if delta > 0 and T < delta / 50, exp is negligible
                    if T > 0.01:
                        r = np.random.random()
                        if r < np.exp(-delta / T):
                            colors[i] = new_val
                            current_score += delta
                            
            T *= cooling_rate
            
    return [int(c) for c in best_colors]
