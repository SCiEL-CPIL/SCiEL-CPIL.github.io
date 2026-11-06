import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try to find a valid coloring for n=34 down to n=1
    for n in range(34, 0, -1):
        if time.time() > deadline - 15:
            break
        success, colors = find_valid_coloring(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring if we ran out of time
    # n=9 from seed is valid.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def find_valid_coloring(n, deadline):
    # Precompute structures for fast energy calculation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s as lists of triple indices
    k4s = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[t] for t in combinations(quad, 3)]
        k4s.append(idxs)
        
    k5s = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[t] for t in combinations(quint, 3)]
        k5s.append(idxs)
        
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # For each triple, precompute which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for ki, idxs in enumerate(k4s):
        for idx in idxs:
            triple_to_k4s[idx].append(ki)
            
    for ki, idxs in enumerate(k5s):
        for idx in idxs:
            triple_to_k5s[idx].append(ki)

    # Vectorized local search strategy
    # Represent colors as a numpy array of 0s and 1s
    # We want to minimize:
    # 1. Number of K4s with 4 reds (all 0s)
    # 2. Number of K5s with 10 blues (all 1s)
    
    def count_violations(colors_np):
        # colors_np is a 1D array of 0s (red) and 1s (blue)
        # K4 violation: sum of colors in 4 triples is 0 (all red)
        # K5 violation: sum of colors in 10 triples is 10 (all blue)
        
        # Efficiently compute sums for all K4s and K5s
        # k4s_indices is a list of lists of indices
        # We can create a matrix where rows are K4s and columns are triples, but that's huge.
        # Instead, we iterate or use sparse logic.
        
        # For n=34, num_k4s = C(34,4) = 46376, num_k5s = C(34,5) = 278256
        # num_triples = C(34,3) = 5984
        
        # Let's use a helper to compute violations quickly
        red_k4_violations = 0
        for idxs in k4s:
            # If all are 0, it's a violation
            # Using numpy sum is fast for small arrays
            if np.sum(colors_np[idxs]) == 0:
                red_k4_violations += 1
                
        blue_k5_violations = 0
        for idxs in k5s:
            # If all are 1, it's a violation
            # Sum of (1-colors) == 0 means all are 1
            # Or sum(colors) == 10
            if np.sum(colors_np[idxs]) == 10:
                blue_k5_violations += 1
                
        return red_k4_violations, blue_k5_violations

    best_violations = float('inf')
    best_colors = None
    
    # Multiple restarts
    max_restarts = 200
    restart_count = 0
    
    while time.time() < deadline - 5 and restart_count < max_restarts:
        restart_count += 1
        
        # Initialize with bias. 
        # To avoid blue K5, we need some reds. 
        # To avoid red K4, we need some blues.
        # A balanced start is usually good, but maybe slightly more red helps break blue K5s easier?
        # Actually, random is fine.
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Calculate initial violations
        r_v, b_v = count_violations(colors)
        total_v = r_v + b_v
        
        if total_v == 0:
            return True, colors.tolist()
            
        if total_v < best_violations:
            best_violations = total_v
            best_colors = colors.copy()
            
        # Local Search: Greedy / Stochastic Hill Climbing
        # We will iterate until no improvement or time limit.
        # To speed up, we can pre-calculate the "impact" of flipping each triple.
        # However, impact depends on current state.
        
        # Strategy: Pick a random triple, calculate delta, accept if better.
        # To make it faster, we can try to fix violations directly.
        
        no_improvement = 0
        max_no_improvement = 5000
        temperature = 1.0
        cooling = 0.999
        
        # Pre-compute indices for faster access? 
        # The bottleneck is count_violations if called every step.
        # We should update violations incrementally.
        
        # Let's maintain r_counts and b_counts for each K4 and K5.
        k4_red_counts = np.zeros(num_k4s, dtype=np.int8) # 0-4
        k5_blue_counts = np.zeros(num_k5s, dtype=np.int8) # 0-10
        
        # Initialize counts
        for ki, idxs in enumerate(k4s):
            k4_red_counts[ki] = np.sum(colors[idxs] == 0)
            
        for ki, idxs in enumerate(k5s):
            k5_blue_counts[ki] = np.sum(colors[idxs] == 1)
            
        # Current violations
        r_v = np.sum(k4_red_counts == 4)
        b_v = np.sum(k5_blue_counts == 10)
        total_v = r_v + b_v
        
        if total_v == 0:
            return True, colors.tolist()
            
        if total_v < best_violations:
            best_violations = total_v
            best_colors = colors.copy()
            
        while no_improvement < max_no_improvement and time.time() < deadline - 5:
            # Pick a random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta in violations
            delta_r_v = 0
            delta_b_v = 0
            
            # Update K4s
            for ki in triple_to_k4s[i]:
                old_red = k4_red_counts[ki]
                if old_color == 0:
                    # Red -> Blue, red count decreases
                    new_red = old_red - 1
                    if old_red == 4: delta_r_v -= 1
                    if new_red == 4: delta_r_v += 1
                    k4_red_counts[ki] = new_red
                else:
                    # Blue -> Red, red count increases
                    new_red = old_red + 1
                    if old_red == 4: delta_r_v -= 1
                    if new_red == 4: delta_r_v += 1
                    k4_red_counts[ki] = new_red
                    
            # Update K5s
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_counts[ki]
                if old_color == 1:
                    # Blue -> Red, blue count decreases
                    new_blue = old_blue - 1
                    if old_blue == 10: delta_b_v -= 1
                    if new_blue == 10: delta_b_v += 1
                    k5_blue_counts[ki] = new_blue
                else:
                    # Red -> Blue, blue count increases
                    new_blue = old_blue + 1
                    if old_blue == 10: delta_b_v -= 1
                    if new_blue == 10: delta_b_v += 1
                    k5_blue_counts[ki] = new_blue
            
            delta_total = delta_r_v + delta_b_v
            
            # Acceptance criterion: Simulated Annealing
            # If delta_total < 0, always accept.
            # If delta_total > 0, accept with prob exp(-delta_total / T)
            # If delta_total == 0, accept with prob 0.5 (to explore)
            
            accepted = False
            if delta_total < 0:
                accepted = True
            elif delta_total == 0:
                if random.random() < 0.5:
                    accepted = True
            else:
                if random.random() < math.exp(-delta_total / max(temperature, 0.01)):
                    accepted = True
            
            if accepted:
                colors[i] = new_color
                r_v += delta_r_v
                b_v += delta_b_v
                total_v += delta_total
                
                if total_v == 0:
                    return True, colors.tolist()
                    
                if total_v < best_violations:
                    best_violations = total_v
                    best_colors = colors.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                # Revert changes
                for ki in triple_to_k4s[i]:
                    if old_color == 0:
                        k4_red_counts[ki] += 1
                    else:
                        k4_red_counts[ki] -= 1
                for ki in triple_to_k5s[i]:
                    if old_color == 1:
                        k5_blue_counts[ki] += 1
                    else:
                        k5_blue_counts[ki] -= 1
                        
                no_improvement += 1
                
            temperature *= cooling
            
        # End of local search for this restart
        
    # If we didn't find a perfect solution, return the best one found?
    # The task requires a VALID coloring. If best_violations > 0, we can't return it as valid.
    # But the evaluator checks validity. If invalid, score 0.
    # So we must return a valid one or a small fallback.
    
    if best_colors is not None and best_violations == 0:
        return True, best_colors.tolist()
        
    return False, None
