import numpy as np
import time
import random

def construct():
    n = 256
    # 1. Generate all APs
    aps = []
    # Pre-calculate indices for each AP
    # Using 0-based indices for numpy
    for d in range(1, n + 1):
        for a in range(0, n):
            # a is start index (0-based)
            # max length
            max_k = (n - 1 - a) // d + 1
            for k in range(1, max_k + 1):
                # Generate indices
                # a, a+d, ..., a+(k-1)d
                idxs = [a + j * d for j in range(k)]
                aps.append(idxs)
    
    M = len(aps)
    
    # 2. Build reverse index: list of AP indices for each position
    aps_by_pos = [[] for _ in range(n)]
    for ap_idx, ap_indices in enumerate(aps):
        for pos in ap_indices:
            aps_by_pos[pos].append(ap_idx)
    
    # Convert to numpy arrays for faster access?
    # aps_by_pos is list of lists of ints.
    # We can keep it as is, or convert to arrays.
    
    # 3. Greedy Initialization
    seq = np.zeros(n, dtype=int)
    sums = np.zeros(M, dtype=int)
    
    # Greedy step
    for i in range(n):
        # Try 1
        # Temporarily update sums for pos i
        indices_i = aps_by_pos[i]
        old_sums_copy = sums[indices_i].copy()
        
        # Case 1: s_i = 1
        sums[indices_i] += 1
        max1 = np.max(np.abs(sums))
        
        # Case 2: s_i = -1
        sums[indices_i] = old_sums_copy - 1
        max2 = np.max(np.abs(sums))
        
        # Choose best
        if max1 <= max2:
            seq[i] = 1
            sums[indices_i] = old_sums_copy + 1
            current_max = max1
        else:
            seq[i] = -1
            sums[indices_i] = old_sums_copy - 1
            current_max = max2
            
    best_seq = seq.copy()
    best_max = current_max
    
    # 4. Local Search
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        indices_i = aps_by_pos[i]
        
        # Current sum of s_i is seq[i]
        # Flip means new s_i = -seq[i]
        # Delta in sums = -2 * seq[i]
        delta = -2 * seq[i]
        
        # Calculate new max if we flip
        # We need to update sums, get max, then revert if rejected
        # To avoid copying whole sums array, we can just update the slice
        old_slice = sums[indices_i].copy()
        sums[indices_i] += delta
        
        new_max = np.max(np.abs(sums))
        
        if new_max < current_max:
            # Accept
            seq[i] = -seq[i]
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_seq = seq.copy()
        elif new_max == current_max:
            # Accept to move on plateau
            seq[i] = -seq[i]
            # current_max unchanged
        else:
            # Reject, revert sums
            sums[indices_i] = old_slice
            
    return best_seq.tolist()
