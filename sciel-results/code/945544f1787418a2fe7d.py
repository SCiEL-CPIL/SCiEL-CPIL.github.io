import numpy as np
import random
import math

def construct():
    n = 256
    # 1. Generate all APs
    ap_indices = []
    ap_index_of = [set() for _ in range(n)]
    
    for d in range(1, n + 1):
        for a in range(n):
            k_max = 1 + (n - 1 - a) // d
            for k in range(1, k_max + 1):
                # Generate indices
                # range(a, a + k * d, d)
                # This creates a list.
                idxs = list(range(a, a + k * d, d))
                ap_id = len(ap_indices)
                ap_indices.append(idxs)
                for idx in idxs:
                    ap_index_of[idx].add(ap_id)
    
    num_aps = len(ap_indices)
    ap_index_of_lists = [list(s) for s in ap_index_of]
    
    # 2. Initialize signs with Thue-Morse
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=int)
    
    # 3. Initial sums
    current_sums = np.zeros(num_aps, dtype=int)
    for ap_id, idxs in enumerate(ap_indices):
        current_sums[ap_id] = np.sum(signs[idxs])
        
    # 4. Histogram
    hist = np.zeros(n + 1, dtype=int)
    for s in current_sums:
        hist[abs(s)] += 1
        
    current_max = n
    while current_max >= 0 and hist[current_max] == 0:
        current_max -= 1
        
    # 5. Simulated Annealing
    T = 1.0
    cooling = 0.9999
    steps = 500000
    
    rand_int = random.randint
    rand_rand = random.random
    exp = math.exp
    
    for step in range(steps):
        i = rand_int(0, n - 1)
        old_sign = signs[i]
        new_sign = -old_sign
        delta = -2 * old_sign
        
        affected_aps = ap_index_of_lists[i]
        
        new_max_affected = 0
        changes = []
        
        for ap_id in affected_aps:
            old_sum = current_sums[ap_id]
            old_abs = abs(old_sum)
            new_sum = old_sum + delta
            new_abs = abs(new_sum)
            
            current_sums[ap_id] = new_sum
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            changes.append((ap_id, old_sum, old_abs, new_abs))
            
            if new_abs > new_max_affected:
                new_max_affected = new_abs
        
        # Find global new max
        # Max is max(new_max_affected, max_unaffected)
        # max_unaffected is the largest v such that hist[v] > 0.
        # We know hist changed only for affected APs.
        
        # Start from max(current_max, new_max_affected)
        candidate_max = max(current_max, new_max_affected)
        while candidate_max >= 0 and hist[candidate_max] == 0:
            candidate_max -= 1
            
        if candidate_max < 0:
            candidate_max = 0
            
        accept = False
        if candidate_max < current_max:
            accept = True
        elif candidate_max == current_max:
            # Accept to allow exploration
            accept = True 
        else:
            delta_e = candidate_max - current_max
            if rand_rand() < exp(-delta_e / T):
                accept = True
                
        if accept:
            signs[i] = new_sign
            current_max = candidate_max
        else:
            # Rollback
            for ap_id, old_sum, old_abs, new_abs in changes:
                current_sums[ap_id] = old_sum
                hist[new_abs] -= 1
                hist[old_abs] += 1
            
        T *= cooling
        
        if T < 0.001:
            break

    return signs.tolist()
