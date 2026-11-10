import numpy as np
import random
import math

def construct():
    n = 256
    
    # 1. Generate all arithmetic progressions
    all_aps = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for d in range(1, n + 1):
        for a in range(n):
            max_k = (n - 1 - a) // d + 1
            for k in range(1, max_k + 1):
                indices = []
                for j in range(k):
                    idx = a + j * d
                    indices.append(idx)
                    index_to_aps[idx].append(ap_id)
                all_aps.append(indices)
                ap_id += 1
                
    num_aps = ap_id
    
    # Convert index_to_aps to list of NumPy arrays for fast vectorized updates
    index_to_aps_np = [np.array(lst, dtype=np.int32) for lst in index_to_aps]
    
    # 2. Initialize
    signs = np.random.choice([-1, 1], size=n)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # Calculate initial sums
    for ap_id_idx, indices in enumerate(all_aps):
        s = 0
        for idx in indices:
            s += signs[idx]
        ap_sums[ap_id_idx] = s
        
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # 3. Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    steps = 100000 # Can increase now that update is vectorized
    
    for step in range(steps):
        if T < min_T:
            break
            
        i = np.random.randint(0, n)
        old_sign = signs[i]
        new_sign = -old_sign
        delta = new_sign - old_sign
        
        # Update ap_sums using vectorized operation
        if len(index_to_aps_np[i]) > 0:
            ap_sums[index_to_aps_np[i]] += delta
            
        new_max = np.max(np.abs(ap_sums))
        
        if new_max < current_max:
            current_max = new_max
            signs[i] = new_sign
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            delta_E = new_max - current_max
            if delta_E == 0:
                signs[i] = new_sign
                current_max = new_max
            else:
                prob = math.exp(-delta_E / T)
                if random.random() < prob:
                    signs[i] = new_sign
                    current_max = new_max
                else:
                    if len(index_to_aps_np[i]) > 0:
                        ap_sums[index_to_aps_np[i]] -= delta
                    
        T *= cooling
        
    return list(best_signs)
