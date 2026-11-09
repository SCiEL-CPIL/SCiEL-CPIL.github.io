"""Improved program for low discrepancy on arithmetic progressions using histogram tracking."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 50  # safety margin before 60s kill

    # Precompute all arithmetic progressions
    # An arithmetic progression is defined by (start, step, length)
    # Indices: start, start+step, ..., start+(length-1)*step
    # All indices must be in [0, n-1]
    
    # aps_by_index[i] will contain the IDs of all APs that include index i
    aps_by_index = [[] for _ in range(n)]
    ap_indices = [] # List of lists of indices, for potential debugging or alternative use
    
    ap_id = 0
    for step in range(1, n):
        for start in range(step):
            # Max length such that start + (L-1)*step < n
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                indices = tuple(start + k * step for k in range(length))
                ap_indices.append(indices)
                for idx in indices:
                    aps_by_index[idx].append(ap_id)
                ap_id += 1
    
    num_aps = len(ap_indices)
    
    # Convert aps_by_index to numpy arrays for fast indexing
    aps_by_index_np = [np.array(lst, dtype=np.int32) for lst in aps_by_index]
    
    # Max possible absolute sum is n (256)
    max_possible_sum = n
    hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
    
    def get_max_from_hist(h):
        # Find the largest index k where h[k] > 0
        # Since h is small, we can just iterate or use nonzero
        non_zero = np.nonzero(h)[0]
        if len(non_zero) == 0:
            return 0
        return non_zero[-1]

    best_signs = None
    best_max = max_possible_sum + 1
    
    rng = np.random.default_rng(42)
    
    # We will perform multiple random restarts
    # Each restart: initialize randomly, then hill climb until stuck or time up
    # To maximize chance, we run restarts until deadline
    
    while time.time() < deadline:
        # 1. Initialize random signs
        signs = rng.choice([-1, 1], size=n).astype(np.int8)
        
        # 2. Initialize sums and histogram
        sums = np.zeros(num_aps, dtype=np.int16)
        hist[:] = 0
        
        for i in range(n):
            s = signs[i]
            ids = aps_by_index_np[i]
            sums[ids] += s
            
        # Populate histogram
        abs_sums = np.abs(sums)
        np.add.at(hist, abs_sums, 1)
        
        current_max = get_max_from_hist(hist)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            
        # 3. Hill Climbing
        no_improve_count = 0
        max_no_improve = 2000 # Restart if stuck for this many flips
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break # Trigger restart
                
            # Pick a random index to flip
            i = rng.integers(0, n)
            
            # Get APs containing i
            ids = aps_by_index_np[i]
            
            # Save old values for histogram update
            old_vals = sums[ids]
            old_abs = np.abs(old_vals)
            
            # Apply flip
            delta = -2 * signs[i]
            sums[ids] += delta
            
            # Update histogram
            new_vals = sums[ids]
            new_abs = np.abs(new_vals)
            
            # Subtract old counts
            np.add.at(hist, old_abs, -1)
            # Add new counts
            np.add.at(hist, new_abs, 1)
            
            # Find new max
            new_max = get_max_from_hist(hist)
            
            # Accept if better
            if new_max < current_max:
                signs[i] = -signs[i]
                current_max = new_max
                no_improve_count = 0
                
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                sums[ids] -= delta
                # Revert histogram
                np.add.at(hist, new_abs, -1)
                np.add.at(hist, old_abs, 1)
                no_improve_count += 1
                
    if best_signs is None:
        # Fallback
        return [1] * n
        
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
