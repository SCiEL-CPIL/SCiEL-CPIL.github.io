"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending
    divisors_desc = sorted(divisors, reverse=True)
    
    covered_count = np.zeros(base, dtype=np.int32)
    classes = []
    residues = []
    
    # Greedy Phase: Large moduli first
    for m in divisors_desc:
        best_r = 0
        best_gain = -1
        
        for r in range(m):
            indices = np.arange(r, base, m)
            gain = np.sum(covered_count[indices] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        residues.append(best_r)
        indices = np.arange(best_r, base, m)
        covered_count[indices] += 1
        
    current_uncovered = np.sum(covered_count == 0)
    best_uncovered = current_uncovered
    best_residues = residues.copy()
    
    # Simulated Annealing Phase
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer for overhead
    T = 10.0
    T_min = 0.01
    n_divs = len(divisors)
    
    # Calculate cooling rate for ~50000 iterations
    # T * r^N = T_min  =>  r = (T_min/T)^(1/N)
    target_iters = 50000
    cooling_rate = (T_min / T) ** (1.0 / target_iters)
    
    iter_count = 0
    while time.time() - start_time < time_limit and T > T_min:
        # Pick random class
        i = random.randint(0, n_divs - 1)
        m = classes[i][0]
        r_old = residues[i]
        
        # Pick random new residue
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            continue
            
        old_idx = np.arange(r_old, base, m)
        new_idx = np.arange(r_new, base, m)
        
        # Apply move
        covered_count[old_idx] -= 1
        covered_count[new_idx] += 1
        
        new_uncovered = np.sum(covered_count == 0)
        delta = new_uncovered - current_uncovered
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_uncovered = new_uncovered
            residues[i] = r_new
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues.copy()
        else:
            # Revert
            covered_count[new_idx] -= 1
            covered_count[old_idx] += 1
            
        T *= cooling_rate
        iter_count += 1
        
    # Reconstruct best classes
    final_classes = []
    for i, m in enumerate([c[0] for c in classes]):
        final_classes.append([m, best_residues[i]])
        
    return final_classes
# EVOLVE-BLOCK-END
