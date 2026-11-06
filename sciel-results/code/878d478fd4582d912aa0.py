import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. Generate all arithmetic progressions
    # AP is defined by start, step, length
    # We store the list of indices for each AP
    # To save memory and time, we just store the list of indices
    progs = []
    for s in range(1, n + 1):
        for a in range(n):
            # indices a, a+s, ... < n
            # Using range
            idx = range(a, n, s)
            if len(idx) > 0:
                progs.append(idx)
                
    # Convert to list of lists for easier indexing? 
    # NumPy needs arrays.
    # Let's store as list of arrays
    prog_arrays = [np.array(p) for p in progs]
    num_progs = len(prog_arrays)
    
    # 2. Precompute which APs contain which index
    # aps_containing[i] is a list of indices into prog_arrays
    aps_containing = [[] for _ in range(n)]
    for p_idx, p in enumerate(prog_arrays):
        for idx in p:
            aps_containing[idx].append(p_idx)
            
    # Convert to arrays for NumPy fancy indexing
    aps_containing_arrays = [np.array(lst) for lst in aps_containing]
    
    # 3. Initialize signs
    # Thue-Morse sequence
    def thue_morse(n):
        return 1 if bin(n).count('1') % 2 == 0 else -1
        
    signs = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # 4. Compute initial sums
    sums = np.zeros(num_progs, dtype=np.int32)
    for p_idx, p in enumerate(prog_arrays):
        sums[p_idx] = np.sum(signs[p])
        
    current_max = np.max(np.abs(sums))
    
    # 5. Local Search
    # We will do passes until no improvement or time limit
    # To avoid getting stuck, we can also try random restarts if stuck for a while
    # But simple hill climbing usually works well for this size.
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
                
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            
            # Update sums
            # delta = -2 * old_sign
            sums[aps_containing_arrays[i]] -= 2 * old_sign
            
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                signs[i] = old_sign
                sums[aps_containing_arrays[i]] += 2 * old_sign
                
        if not improved:
            # Stuck. Try a random flip to escape local minimum?
            # Or just break if we've been stuck for a while?
            # Let's try a few random flips to perturb
            for _ in range(10):
                if time.time() > deadline:
                    break
                i = np.random.randint(n)
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                sums[aps_containing_arrays[i]] -= 2 * old_sign
                # Don't check max, just accept perturbation
            # Then continue searching
            
    # Return as list of ints
    return [int(s) for s in signs]
