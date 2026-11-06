import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort()
    
    # Precompute indices
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            idx = np.arange(r, base, m)
            res_indices.append(idx)
        indices_by_m[m] = res_indices

    best_uncovered = base
    best_classes = []
    
    # Run multiple greedy strategies
    num_runs = 20 # Adjust based on time
    
    for run in range(num_runs):
        if time.time() > deadline - 5: # Check time
            break
            
        covered = np.zeros(base, dtype=np.uint8)
        available_moduli = set(divisors)
        classes = []
        
        while available_moduli:
            if time.time() > deadline - 2:
                break
                
            best_gain = -1
            best_m = None
            best_r = None
            best_indices = None
            
            # Add noise to gain to randomize selection among similar options
            noise_scale = 0.1 if run > 0 else 0.0
            
            for m in available_moduli:
                res_indices_list = indices_by_m[m]
                for r in range(m):
                    idx = res_indices_list[r]
                    gain = np.sum(covered[idx] == 0)
                    if noise_scale > 0:
                        gain += random.uniform(0, noise_scale)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
                        best_indices = idx
            
            if best_gain <= 0:
                break
                
            # Commit
            classes.append([best_m, best_r])
            covered[best_indices] = 1
            available_moduli.remove(best_m)
            
            current_uncovered = np.sum(covered == 0)
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
                
            if best_uncovered == 0:
                return best_classes

    return best_classes
