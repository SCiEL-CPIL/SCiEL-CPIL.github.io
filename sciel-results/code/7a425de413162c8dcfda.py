import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85
    
    # Generate all APs with length >= 2
    ap_indices = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                ap_indices.append(seq)
    
    m = len(ap_indices)
    
    # Map index to list of AP IDs
    index_to_aps = [[] for _ in range(n)]
    for ap_id, ap in enumerate(ap_indices):
        for idx in ap:
            index_to_aps[idx].append(ap_id)
    
    best_signs = None
    best_max = float('inf')
    
    # Initializations to try
    inits = []
    
    # 1. Thue-Morse
    tm = [1]
    while len(tm) < n:
        tm += [-x for x in tm]
    inits.append(np.array(tm[:n], dtype=np.int8))
    
    # 2. Random
    inits.append(np.array([1 if random.random() > 0.5 else -1 for _ in range(n)], dtype=np.int8))
    
    # 3. Greedy (minimize prefix sum magnitude)
    greedy = np.zeros(n, dtype=np.int8)
    prefix_sum = 0
    for i in range(n):
        # Try 1 and -1
        # If prefix_sum > 0, prefer -1
        # If prefix_sum < 0, prefer 1
        if prefix_sum > 0:
            greedy[i] = -1
            prefix_sum -= 1
        elif prefix_sum < 0:
            greedy[i] = 1
            prefix_sum += 1
        else:
            greedy[i] = 1 if random.random() > 0.5 else -1
            prefix_sum += greedy[i]
    inits.append(greedy)
    
    for signs in inits:
        if time.time() > deadline - 10:
            break
            
        ap_sums = np.zeros(m, dtype=np.int16)
        for i in range(n):
            s = signs[i]
            for ap_id in index_to_aps[i]:
                ap_sums[ap_id] += s
                
        current_max = np.max(np.abs(ap_sums))
        
        # Local Search
        T = 1.0
        cooling = 0.9995
        min_T = 0.005
        
        # Track best in this run
        run_best_max = current_max
        run_best_signs = signs.copy()
        
        while time.time() < deadline - 2:
            if T < min_T:
                break
                
            # Find APs with |sum| > 1
            # If none, we have a perfect solution (max sum 1)
            violating_aps = np.where(np.abs(ap_sums) > 1)[0]
            
            if len(violating_aps) == 0:
                # Max sum is 1
                break
                
            # Pick a random violating AP
            ap_id = violating_aps[random.randint(0, len(violating_aps)-1)]
            ap = ap_indices[ap_id]
            
            # Pick a random index in this AP
            i = ap[random.randint(0, len(ap)-1)]
            
            old_sign = signs[i]
            new_sign = -old_sign
            delta = -2 * old_sign
            
            affected_aps = index_to_aps[i]
            
            # Update sums
            old_sums = ap_sums[affected_aps].copy()
            ap_sums[affected_aps] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            # Acceptance
            if new_max < current_max:
                pass
            else:
                # Calculate delta E
                # We care about the max sum
                # If new_max > current_max, we might reject
                # Simple SA: accept with prob exp(-(new_max - current_max)/T)
                if new_max > current_max:
                    prob = np.exp(-(new_max - current_max) / T)
                    if random.random() > prob:
                        ap_sums[affected_aps] = old_sums
                        T *= cooling
                        continue
                else:
                    # new_max <= current_max, but maybe not strictly less?
                    # If new_max == current_max, we might still accept to move around
                    # Let's accept if it reduces the number of violating APs
                    # Count violations before and after? Expensive.
                    # Just accept if new_max <= current_max
                    pass
            
            signs[i] = new_sign
            current_max = new_max
            T *= cooling
            
            if current_max < run_best_max:
                run_best_max = current_max
                run_best_signs = signs.copy()
                
        if run_best_max < best_max:
            best_max = run_best_max
            best_signs = run_best_signs
            
    if best_signs is None:
        return [1] * n
        
    return best_signs.tolist()
