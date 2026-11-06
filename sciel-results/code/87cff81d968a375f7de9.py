import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Store as 0-indexed
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Build incidence: for each position i, which APs contain it
    # And for each AP, what are the other 3 positions
    ap_list = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, list of (ap_index, position_in_ap)
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for k in range(4):
            pos = ap_list[ap_idx, k]
            pos_to_aps[pos].append((ap_idx, k))
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is array of 0/1."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
                count += 1
        return count
    
    def delta_for_flip(coloring, pos):
        """Compute change in mono count if we flip coloring[pos]."""
        delta = 0
        for (ap_idx, k) in pos_to_aps[pos]:
            ap = ap_list[ap_idx]
            # Get the other 3 positions
            others = [ap[j] for j in range(4) if j != k]
            cur_val = coloring[pos]
            other_vals = [coloring[o] for o in others]
            
            # Before flip: all four equal?
            before_mono = 1
            for ov in other_vals:
                if ov != cur_val:
                    before_mono = 0
                    break
            
            # After flip: all four equal?
            new_val = 1 - cur_val
            after_mono = 1
            for ov in other_vals:
                if ov != new_val:
                    after_mono = 0
                    break
            
            delta += after_mono - before_mono
        return delta
    
    # Initialize with Thue-Morse sequence (known to have few 4-APs)
    def thue_morse(n):
        """Return Thue-Morse sequence of length n."""
        t = [0]
        for _ in range(n.bit_length()):
            t = t + [1 - x for x in t]
        return t[:n]
    
    coloring = np.array(thue_morse(N), dtype=np.int8)
    best_coloring = coloring.copy()
    best_count = count_mono(best_coloring)
    
    # Coordinate descent with restarts
    # Phase 1: Greedy coordinate descent
    def coordinate_descent(coloring, max_rounds=200):
        """Perform coordinate descent. Returns improved coloring and count."""
        coloring = coloring.copy()
        count = count_mono(coloring)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for pos in range(N):
                if time.time() > deadline - 5.0:
                    break
                d = delta_for_flip(coloring, pos)
                if d < 0:
                    coloring[pos] = 1 - coloring[pos]
                    count += d
                    improved = True
            if not improved:
                break
        return coloring, count
    
    # Run coordinate descent from initial
    coloring, count = coordinate_descent(coloring)
    if count < best_count:
        best_count = count
        best_coloring = coloring.copy()
    
    # Phase 2: Simulated annealing with multiple restarts
    def simulated_annealing(initial_coloring, t_start=1.0, t_end=0.01, num_iters=50000):
        """Simulated annealing. Returns best coloring found and its count."""
        coloring = initial_coloring.copy()
        count = count_mono(coloring)
        best_local = coloring.copy()
        best_local_count = count
        
        # Precompute for speed
        for it in range(num_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick random position
            pos = np.random.randint(N)
            d = delta_for_flip(coloring, pos)
            
            # Accept if improvement or with probability exp(-d/T)
            T = t_start * (t_end / t_start) ** (it / num_iters)
            if d < 0 or np.random.random() < np.exp(-d / T):
                coloring[pos] = 1 - coloring[pos]
                count += d
                if count < best_local_count:
                    best_local_count = count
                    best_local = coloring.copy()
        
        return best_local, best_local_count
    
    # Run several SA restarts from different starting points
    num_restarts = 0
    while time.time() < deadline - 10.0:
        # Create diverse starting points
        if num_restarts == 0:
            start_coloring = best_coloring.copy()
        elif num_restarts % 2 == 1:
            # Random perturbation of best
            start_coloring = best_coloring.copy()
            num_flips = np.random.randint(10, 50)
            flip_positions = np.random.choice(N, size=num_flips, replace=False)
            start_coloring[flip_positions] = 1 - start_coloring[flip_positions]
        else:
            # Fresh random
            start_coloring = np.random.randint(0, 2, size=N, dtype=np.int8)
        
        sa_coloring, sa_count = simulated_annealing(start_coloring, t_start=1.0, t_end=0.01, num_iters=30000)
        
        if sa_count < best_count:
            best_count = sa_count
            best_coloring = sa_coloring.copy()
        
        # Follow up with coordinate descent
        cd_coloring, cd_count = coordinate_descent(sa_coloring, max_rounds=100)
        if cd_count < best_count:
            best_count = cd_count
            best_coloring = cd_coloring.copy()
        
        num_restarts += 1
    
    # Final polish: coordinate descent on best
    cd_coloring, cd_count = coordinate_descent(best_coloring, max_rounds=500)
    if cd_count < best_count:
        best_count = cd_count
        best_coloring = cd_coloring.copy()
    
    # Convert to list of ints
    return [int(c) for c in best_coloring]
