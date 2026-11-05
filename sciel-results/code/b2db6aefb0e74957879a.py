import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin for 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with indices 0..255
    ap_list = []
    for d in range(1, n // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # Helper: count mono APs given a coloring (0/1 array)
    def count_mono(coloring):
        # coloring is np array of 0/1, length n
        # For each AP, check if all 4 are same
        c = coloring[ap_array]  # shape (num_aps, 4)
        # mono if all equal: c[:,0]==c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
    
    # Thue-Morse initialization
    def thue_morse(n):
        # Thue-Morse: t(i) = parity of popcount(i)
        tm = np.zeros(n, dtype=np.int32)
        for i in range(n):
            # Count bits
            x = i
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            tm[i] = cnt % 2
        return tm
    
    # Vectorized delta computation: for flipping bit i, compute change in mono count
    def compute_all_deltas(coloring):
        """
        For each position i (0..n-1), compute the delta in mono count if we flip coloring[i].
        Uses the fact that flipping i affects all APs containing i.
        """
        # For each AP, determine its current mono status
        c = coloring[ap_array]  # (num_aps, 4)
        is_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        # For flipping position i, we need to check APs where i appears
        # For each AP j and position p in the AP, if we flip ap_array[j,p], 
        # the AP changes mono status only if it was mono or becomes mono
        # 
        # More efficient: for each AP, if we flip one of its elements,
        # the new mono status depends on the other 3 elements
        # 
        # Let's compute for each AP: 
        #   - current mono?
        #   - for each of the 4 positions, if flipped, would it become mono?
        
        # An AP is mono iff all 4 equal.
        # If we flip position p in AP j, the new values are:
        #   c[j, :p] unchanged, c[j,p] flipped, c[j, p+1:] unchanged
        # New mono iff all 4 new values equal.
        # This happens iff the 3 unflipped values are all equal to the flipped value's new value.
        # i.e., the 3 unflipped values are all equal, AND they equal 1 - c[j,p]
        
        deltas = np.zeros(n, dtype=np.int32)
        
        for p in range(4):
            # For APs where position p is the one being flipped
            # The other 3 positions are the other indices
            other = [q for q in range(4) if q != p]
            other_vals = c[:, other]  # (num_aps, 3)
            # The 3 other values are all equal?
            others_equal = (other_vals[:, 0] == other_vals[:, 1]) & (other_vals[:, 1] == other_vals[:, 2])
            # The flipped value's new value is 1 - c[:, p]
            # For the AP to become mono after flip: others_equal AND others[0] == 1 - c[:, p]
            would_become_mono = others_equal & (other_vals[:, 0] == 1 - c[:, p])
            
            # For each AP j, the position being flipped is ap_array[j, p]
            pos = ap_array[:, p]  # (num_aps,)
            
            # For each AP j:
            #   if it was mono and flipping makes it non-mono: delta -= 1
            #   if it was non-mono and flipping makes it mono: delta += 1
            #   Note: if it was mono, flipping one element always makes it non-mono (since the flipped one differs)
            #   So: delta for AP j = (would_become_mono ? +1 : 0) - (is_mono ? 1 : 0)
            
            delta_j = np.where(would_become_mono, 1, 0) - np.where(is_mono, 1, 0)
            
            # Accumulate into deltas at position pos
            np.add.at(deltas, pos, delta_j)
        
        return deltas
    
    # Local search: greedily flip positions that reduce mono count
    def local_search(coloring, max_sweeps=2000):
        current = coloring.copy()
        current_count = count_mono(current)
        improved = True
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline:
            improved = False
            sweep += 1
            deltas = compute_all_deltas(current)
            # Find positions with negative delta (improvement)
            best_pos = np.argmin(deltas)
            if deltas[best_pos] < 0:
                current[best_pos] = 1 - current[best_pos]
                current_count += deltas[best_pos]
                improved = True
            # Also try multiple flips per sweep for faster convergence
            # Sort by delta and flip all improving ones
        return current, current_count
    
    # Simulated annealing step
    def simulated_annealing(coloring, start_temp=5.0, end_temp=0.1, steps=5000):
        current = coloring.copy()
        current_count = count_mono(current)
        best = current.copy()
        best_count = current_count
        
        for step in range(steps):
            if time.time() > deadline:
                break
            temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
            # Pick random position to flip
            pos = np.random.randint(n)
            # Compute delta for this single flip
            c = current[ap_array]
            is_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
            # Find APs containing pos
            mask = (ap_array == pos).any(axis=1)
            if not np.any(mask):
                delta = 0
            else:
                affected = ap_array[mask]
                affected_mono = is_mono[mask]
                # For each affected AP, check if flipping pos changes mono status
                new_delta = 0
                for j in range(affected.shape[0]):
                    ap = affected[j]
                    old_mono = affected_mono[j]
                    # Find which position in the AP is pos
                    p_idx = np.where(ap == pos)[0][0]
                    other = [q for q in range(4) if q != p_idx]
                    other_vals = current[ap[other]]
                    others_equal = all(other_vals[k] == other_vals[0] for k in range(1, 3))
                    would_become_mono = others_equal and (other_vals[0] == 1 - current[pos])
                    if old_mono and not would_become_mono:
                        new_delta -= 1
                    elif not old_mono and would_become_mono:
                        new_delta += 1
                delta = new_delta
            
            if delta <= 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                current[pos] = 1 - current[pos]
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best = current.copy()
        
        return best, best_count
    
    # Initialize with Thue-Morse
    coloring = thue_morse(n)
    best_coloring = coloring.copy()
    best_count = count_mono(coloring)
    
    # Multiple restarts with local search + SA
    rng = np.random.RandomState(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            start = coloring.copy()
        else:
            # Perturb the best solution
            start = best_coloring.copy()
            # Flip a random subset
            num_flips = rng.randint(5, 20)
            flip_positions = rng.choice(n, size=num_flips, replace=False)
            start[flip_positions] = 1 - start[flip_positions]
        
        # Local search
        ls_result, ls_count = local_search(start, max_sweeps=3000)
        
        if ls_count < best_count:
            best_count = ls_count
            best_coloring = ls_result.copy()
        
        # Simulated annealing from local search result
        sa_result, sa_count = simulated_annealing(ls_result, start_temp=3.0, end_temp=0.05, steps=3000)
        
        if sa_count < best_count:
            best_count = sa_count
            best_coloring = sa_result.copy()
        
        # Another local search from SA result
        ls2_result, ls2_count = local_search(sa_result, max_sweeps=2000)
        
        if ls2_count < best_count:
            best_count = ls2_count
            best_coloring = ls2_result.copy()
    
    # Final local search
    if time.time() < deadline - 2:
        final_result, final_count = local_search(best_coloring, max_sweeps=5000)
        if final_count < best_count:
            best_count = final_count
            best_coloring = final_result.copy()
    
    # Return as list of 0/1 for integers 1..256
    return [int(x) for x in best_coloring]
