import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # Leave a safety margin
    
    # Generate all APs within [0, n)
    aps = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                aps.append(seq)
    
    m = len(aps)
    
    # Precompute for each index which APs it belongs to
    index_to_aps = [[] for _ in range(n)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(ap_id)
    
    A = np.zeros((n, m), dtype=np.int8)
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            A[idx, ap_id] = 1
    
    # Greedy initialization: assign signs one by one to minimize local AP conflicts
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(m, dtype=np.int16)
    
    for i in range(n):
        # Try both signs and pick the one that minimizes max AP sum magnitude
        best_sign = 1
        best_max = float('inf')
        for s in [1, -1]:
            # Compute what the max AP sum would be if we set signs[i] = s
            # Only APs containing i are affected
            idxs = np.nonzero(A[i, :])[0]
            # Temporarily add this sign's contribution
            old_sums = ap_sums[idxs].copy()
            ap_sums[idxs] += s
            new_max = np.max(np.abs(ap_sums[idxs]))
            # Revert
            ap_sums[idxs] = old_sums
            
            if new_max < best_max:
                best_max = new_max
                best_sign = s
        
        signs[i] = best_sign
        idxs = np.nonzero(A[i, :])[0]
        ap_sums[idxs] += best_sign
    
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search: Hill Climbing
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2:
                break
            
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            
            idxs = np.nonzero(A[i, :])[0]
            ap_sums[idxs] += 2 * new_sign
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                ap_sums[idxs] -= 2 * new_sign
                signs[i] = old_sign
        
        if improved:
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_signs.tolist()
