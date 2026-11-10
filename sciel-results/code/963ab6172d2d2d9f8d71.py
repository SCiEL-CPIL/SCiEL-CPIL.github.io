import numpy as np
import random
import math
import time

def construct():
    n = 256
    # Precompute APs
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    # Max possible sum is n
    max_val = n
    
    for d in range(1, n):
        for a in range(n):
            idx = a
            current_indices = []
            while idx < n:
                current_indices.append(idx)
                ap_idx = len(aps)
                aps.append(np.array(current_indices, dtype=np.int32))
                for k in current_indices:
                    index_to_aps[k].append(ap_idx)
                idx += d
                
    # Convert index_to_aps to numpy arrays for faster indexing
    index_to_aps_np = [np.array(lst, dtype=np.int32) for lst in index_to_aps]
    
    num_aps = len(aps)
    
    # Initialize signs
    signs = np.random.choice([-1, 1], size=n)
    
    # Initialize sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for i in range(num_aps):
        ap_sums[i] = np.sum(signs[aps[i]])
        
    # Histogram of absolute sums
    hist = np.zeros(max_val + 1, dtype=np.int32)
    for s in ap_sums:
        hist[abs(s)] += 1
        
    def get_max_abs(hist):
        # Find largest index with non-zero count
        # hist is small (257), so linear scan is fine
        for i in range(max_val, -1, -1):
            if hist[i] > 0:
                return i
        return 0
        
    current_max = get_max_abs(hist)
    
    # Simulated Annealing
    T = 10.0
    cool_rate = 0.999
    start_time = time.time()
    time_limit = 50.0 # Leave some buffer
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        old_sign = signs[i]
        delta = -2 * old_sign
        
        ap_indices = index_to_aps_np[i]
        if len(ap_indices) == 0:
            continue
            
        old_sums = ap_sums[ap_indices]
        old_abs = np.abs(old_sums)
        
        # Update histogram for old sums
        # np.unique with return_counts is efficient for small arrays
        # But ap_indices can be ~18000.
        # Let's just loop? No, vectorize.
        # We can use bincount
        hist[old_abs] -= np.bincount(old_abs, minlength=max_val+1)
        
        # Update sums
        new_sums = old_sums + delta
        ap_sums[ap_indices] = new_sums
        
        new_abs = np.abs(new_sums)
        hist[new_abs] += np.bincount(new_abs, minlength=max_val+1)
        
        new_max = get_max_abs(hist)
        
        # Acceptance criterion
        if new_max < current_max:
            signs[i] = -old_sign
            current_max = new_max
        else:
            # Revert
            ap_sums[ap_indices] = old_sums
            # Revert hist
            hist[new_abs] -= np.bincount(new_abs, minlength=max_val+1)
            hist[old_abs] += np.bincount(old_abs, minlength=max_val+1)
            
        T *= cool_rate
        
    # Convert back to list of ints
    return [int(s) for s in signs]
