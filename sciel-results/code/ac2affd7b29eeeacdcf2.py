import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Store as 0-indexed
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_list = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For fast delta computation: for each position, store the AP indices and positions
    # Then precompute the "other positions" as arrays for vectorized ops
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for k in range(4):
            pos = ap_list[ap_idx, k]
            pos_to_aps[pos].append((ap_idx, k))
    
    # Precompute for each position: array of other positions for each AP it's in
    # pos_other_positions[pos] = list of arrays, each of length 3
    pos_other_positions = []
    for pos in range(N):
        others_list = []
        for (ap_idx, k) in pos_to_aps[pos]:
            ap = ap_list[ap_idx]
            others = [ap[j] for j in range(4) if j != k]
            others_list.append(np.array(others, dtype=np.int32))
        pos_other_positions.append(others_list)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is array of 0/1."""
        # Vectorized: check all APs at once
        c = coloring[ap_list]  # shape (num_aps, 4)
        # All four equal: c[:,0]==c[:,1]==c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
    
    def fast_delta(coloring, pos):
        """Compute change in mono count if we flip coloring[pos]. Vectorized."""
        if not pos_to_aps[pos]:
            return 0
        cur_val = coloring[pos]
        new_val = 1 - cur_val
        
        # For each AP containing pos, check if it becomes/becomes mono
        delta = 0
        for others in pos_other_positions[pos]:
            other_vals = coloring[others]
            # Before: all four equal?
            before_mono = 1
            for ov in other_vals:
                if ov != cur_val:
                    before_mono = 0
                    break
            # After: all four equal?
            after_mono = 1
            for ov in other_vals:
                if ov != new_val:
                    after_mono = 0
                    break
            delta += after_mono - before_mono
        return delta
    
    def local_search(coloring, max_rounds=100):
        """Iterated local search with 1-bit flips. Returns improved coloring and count."""
        coloring = coloring.copy()
        count = count_mono(coloring)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try all positions
            for pos in range(N):
                if time.time() > deadline - 5.0:
                    break
                d = fast_delta(coloring, pos)
                if d < 0:
                    coloring[pos] = 1 - coloring[pos]
                    count += d
                    improved = True
            if not improved:
                break
        return coloring, count
    
    def iterated_local_search(initial_coloring, num_iters=100):
        """ILS: perturb, local search, keep best."""
        coloring = initial_coloring.copy()
        count = count_mono(coloring)
        best_local = coloring.copy()
        best_local_count = count
        
        for it in range(num_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb: flip 3-8 random bits
            num_flips = np.random.randint(3, 9)
            flip_positions = np.random.choice(N, size=num_flips, replace=False)
            perturbed = coloring.copy()
            perturbed[flip_positions] = 1 - perturbed[flip_positions]
            
            # Local search on perturbed
            ls_coloring, ls_count = local_search(perturbed, max_rounds=50)
            
            if ls_count < best_local_count:
                best_local_count = ls_count
                best_local = ls_coloring.copy()
                coloring = ls_coloring.copy()
                count = ls_count
        
        return best_local, best_local_count
    
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
    
    # Phase 1: Initial local search
    coloring, count = local_search(coloring, max_rounds=200)
    if count < best_count:
        best_count = count
        best_coloring = coloring.copy()
    
    # Phase 2: Iterated local search from best
    ilc_coloring, ilc_count = iterated_local_search(best_coloring, num_iters=200)
    if ilc_count < best_count:
        best_count = ilc_count
        best_coloring = ilc_coloring.copy()
    
    # Phase 3: More ILS restarts with diverse starting points
    num_restarts = 0
    while time.time() < deadline - 15.0:
        # Create diverse starting points
        if num_restarts == 0:
            start_coloring = best_coloring.copy()
        elif num_restarts % 3 == 1:
            # Medium perturbation of best
            start_coloring = best_coloring.copy()
            num_flips = np.random.randint(10, 30)
            flip_positions = np.random.choice(N, size=num_flips, replace=False)
            start_coloring[flip_positions] = 1 - start_coloring[flip_positions]
        elif num_restarts % 3 == 2:
            # Large perturbation of best
            start_coloring = best_coloring.copy()
            num_flips = np.random.randint(30, 60)
            flip_positions = np.random.choice(N, size=num_flips, replace=False)
            start_coloring[flip_positions] = 1 - start_coloring[flip_positions]
        else:
            # Fresh random
            start_coloring = np.random.randint(0, 2, size=N, dtype=np.int8)
        
        ilc_coloring, ilc_count = iterated_local_search(start_coloring, num_iters=50)
        
        if ilc_count < best_count:
            best_count = ilc_count
            best_coloring = ilc_coloring.copy()
        
        num_restarts += 1
    
    # Final polish: local search on best with more rounds
    cd_coloring, cd_count = local_search(best_coloring, max_rounds=500)
    if cd_count < best_count:
        best_count = cd_count
        best_coloring = cd_coloring.copy()
    
    # Convert to list of ints
    return [int(c) for c in best_coloring]
