import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute combinations
    # We will try n from 34 down to a minimum if we fail, or up if we succeed.
    # Let's start with n=34.
    
    def solve_for_n(n):
        # Returns valid colors list or None
        if n < 4:
            return [0] * (n*(n-1)*(n-2)//6)
        
        num_triples = n * (n - 1) * (n - 2) // 6
        # Map (a,b,c) to index
        # Lexicographic order: 
        # (0,1,2), (0,1,3)...
        # We can generate indices on the fly or precompute mapping.
        # For speed, precompute triple indices for all combinations.
        
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4s = list(itertools.combinations(range(n), 4))
        k5s = list(itertools.combinations(range(n), 5))
        
        # For each triple, which K4s and K5s does it belong to?
        # And which triples are in each K4/K5?
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        k4_triples = []
        for k4 in k4s:
            k4_triple_indices = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        idx = triple_index[t]
                        k4_triple_indices.append(idx)
                        triple_to_k4s[idx].append(len(k4_triples)) # store k4 index
            k4_triples.append(k4_triple_indices)
            
        triple_to_k5s = [[] for _ in range(num_triples)]
        k5_triples = []
        for k5 in k5s:
            k5_triple_indices = []
            for t in itertools.combinations(k5, 3):
                idx = triple_index[t]
                k5_triple_indices.append(idx)
                triple_to_k5s[idx].append(len(k5_triples))
            k5_triples.append(k5_triple_indices)
            
        # Initialize state
        # Random start
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Count violations
        # red_count_in_k4[k] = number of red triples in k4[k]
        red_count_in_k4 = np.zeros(len(k4s), dtype=np.int8)
        for k4_idx, t_indices in enumerate(k4_triples):
            red_count_in_k4[k4_idx] = np.sum(colors[t_indices] == 0)
            
        # blue_count_in_k5[k] = number of blue triples in k5[k]
        blue_count_in_k5 = np.zeros(len(k5s), dtype=np.int8)
        for k5_idx, t_indices in enumerate(k5_triples):
            blue_count_in_k5[k5_idx] = np.sum(colors[t_indices] == 1)
            
        num_red_violations = np.sum(red_count_in_k4 == 4)
        num_blue_violations = np.sum(blue_count_in_k5 == 10)
        total_violations = num_red_violations + num_blue_violations
        
        if total_violations == 0:
            return colors.tolist()
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        # Heuristic: try to flip triples that are in many violations
        # But random is safer for SA
        
        while T > min_T:
            if time.time() > deadline:
                return None # Timeout, caller handles fallback
            
            # Pick a random triple
            t_idx = np.random.randint(num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[t_idx]:
                old_count = red_count_in_k4[k4_idx]
                if old_color == 0: # was red, becomes blue
                    red_count_in_k4[k4_idx] = old_count - 1
                    if old_count == 4:
                        delta_red -= 1 # removed violation
                    # if old_count was 3, it becomes 2, no change
                else: # was blue, becomes red
                    red_count_in_k4[k4_idx] = old_count + 1
                    if old_count == 3:
                        delta_red += 1 # new violation
                    # if old_count was 4, it becomes 5, no change (still violation)
                    
            # Update K5 counts
            for k5_idx in triple_to_k5s[t_idx]:
                old_count = blue_count_in_k5[k5_idx]
                if old_color == 1: # was blue, becomes red
                    blue_count_in_k5[k5_idx] = old_count - 1
                    if old_count == 10:
                        delta_blue -= 1 # removed violation
                else: # was red, becomes blue
                    blue_count_in_k5[k5_idx] = old_count + 1
                    if old_count == 9:
                        delta_blue += 1 # new violation
                        
            delta_total = delta_red + delta_blue
            
            if delta_total < 0:
                # Accept
                colors[t_idx] = new_color
                num_red_violations += delta_red
                num_blue_violations += delta_blue
                total_violations += delta_total
            else:
                # Accept with probability exp(-delta/T)
                if np.random.rand() < np.exp(-delta_total / T):
                    colors[t_idx] = new_color
                    num_red_violations += delta_red
                    num_blue_violations += delta_blue
                    total_violations += delta_total
                    
            T *= cooling
            
            if total_violations == 0:
                return colors.tolist()
                
        return None

    # Try n=34 first
    best_n = 0
    best_colors = None
    
    # We can try increasing n if we succeed, or decreasing if we fail.
    # Since 34 is the known bound, start there.
    # If 34 fails, try 33, etc.
    
    # To maximize n, we should try high n first.
    # If we find a solution for 34, we can try 35.
    
    # Let's try 34.
    target_n = 34
    colors_34 = solve_for_n(target_n)
    if colors_34 is not None:
        best_n = 34
        best_colors = colors_34
        
        # Try 35 if time permits
        if time.time() < deadline - 10:
            colors_35 = solve_for_n(35)
            if colors_35 is not None:
                best_n = 35
                best_colors = colors_35
    
    if best_n == 0:
        # Fallback: try smaller n
        for n in range(33, 1, -1):
            if time.time() > deadline:
                break
            colors_n = solve_for_n(n)
            if colors_n is not None:
                best_n = n
                best_colors = colors_n
                break
        
    if best_n == 0:
        # Absolute fallback: n=1
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}
