import numpy as np
import time
import random

def construct():
    base = 51975
    # Find all divisors of base greater than 1
    divisors = []
    for i in range(2, base + 1):
        if base % i == 0:
            divisors.append(i)
    
    # Precompute index arrays for each modulus and residue
    idx_cache = {}
    for m in divisors:
        idx_cache[m] = []
        for r in range(m):
            idx_cache[m].append(np.arange(r, base, m))
    
    # Initialize counts and score
    counts = np.zeros(base, dtype=np.int16)
    current_score = base  # Initially all uncovered
    
    # Dynamic Greedy Initialization
    # At each step, pick the (m, r) among unassigned moduli that maximally reduces uncovered count
    unassigned = list(divisors)
    assigned = {}
    
    while unassigned:
        best_m = None
        best_r = None
        max_new_cover = -1
        
        for m in unassigned:
            for r in range(m):
                idx = idx_cache[m][r]
                new_cover = np.sum(counts[idx] == 0)
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_m = m
                    best_r = r
        
        # Apply best choice
        assigned[best_m] = best_r
        idx = idx_cache[best_m][best_r]
        counts[idx] += 1
        current_score -= max_new_cover
        unassigned.remove(best_m)
    
    # Local Search (Hill Climbing) with multiple passes
    start_time = time.time()
    deadline = start_time + 145.0
    
    moduli_list = list(divisors)
    best_score = current_score
    best_assigned = dict(assigned)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle moduli to break symmetry
        random.shuffle(moduli_list)
        
        for m in moduli_list:
            if time.time() > deadline:
                break
            
            current_r = assigned[m]
            best_r = current_r
            best_delta = 0  # We want delta < 0 for improvement
            
            old_idx = idx_cache[m][current_r]
            lost = np.sum(counts[old_idx] == 1)
            
            for r in range(m):
                if r == current_r:
                    continue
                
                new_idx = idx_cache[m][r]
                gained = np.sum(counts[new_idx] == 0)
                
                delta = lost - gained
                if delta < best_delta:
                    best_delta = delta
                    best_r = r
            
            if best_r != current_r:
                # Apply move
                old_idx = idx_cache[m][current_r]
                new_idx = idx_cache[m][best_r]
                
                counts[old_idx] -= 1
                counts[new_idx] += 1
                
                current_score += best_delta
                assigned[m] = best_r
                improved = True
                
                # Track best
                if current_score < best_score:
                    best_score = current_score
                    best_assigned = dict(assigned)
    
    # Use the best solution found
    for m in divisors:
        assigned[m] = best_assigned[m]
    
    # Construct result
    classes = []
    for m in divisors:
        classes.append([m, assigned[m]])
        
    return classes
