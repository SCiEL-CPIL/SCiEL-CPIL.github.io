"""Rewritten program for hypergraph Ramsey R(4,5;3) coloring."""
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def solve_n(n, time_budget=10.0):
        """Try to find a valid coloring for n vertices using constraint propagation + local search."""
        if n < 4:
            return None
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = [triple_idx[tri] for tri in combinations(k4, 3)]
            k4_triples.append(ts)
            
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = [triple_idx[tri] for tri in combinations(k5, 3)]
            k5_triples.append(ts)
            
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
        
        # Try multiple initializations
        inits = []
        # Mod 3
        inits.append([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples])
        # Mod 2
        inits.append([0 if (a+b+c) % 2 == 0 else 1 for a,b,c in triples])
        # Random
        inits.append([random.randint(0,1) for _ in range(num_triples)])
        
        best_result = None
        best_violations = float('inf')
        
        for init_colors in inits:
            if time_left() < 5.0:
                break
            
            colors = init_colors[:]
            
            # Initialize counts
            red_count_k4 = np.zeros(num_k4, dtype=np.int32)
            blue_count_k5 = np.zeros(num_k5, dtype=np.int32)
            
            for i, ts in enumerate(k4_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 0:
                        cnt += 1
                red_count_k4[i] = cnt
                
            for i, ts in enumerate(k5_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 1:
                        cnt += 1
                blue_count_k5[i] = cnt
            
            red_violations = int(np.sum(red_count_k4 == 4))
            blue_violations = int(np.sum(blue_count_k5 == 10))
            
            if red_violations == 0 and blue_violations == 0:
                return colors
            
            # Constraint propagation: try to force colors based on near-violations
            # A K4 with 3 red triples has 1 blue triple - if we flip it, we create a violation
            # A K4 with 4 red triples is a violation
            # A K5 with 9 blue triples has 1 red triple - if we flip it, we create a violation
            # A K5 with 10 blue triples is a violation
            
            # Local search with simulated annealing
            T = 1.0
            cooling = 0.9995
            min_T = 0.001
            step = 0
            max_steps = 50000
            
            local_start = time.time()
            while step < max_steps and time_left() > 5.0 and (time.time() - local_start) < time_budget:
                step += 1
                
                # Pick a triple biased towards violations
                # 70% chance: pick from a violating or near-violating structure
                # 30% chance: pick random
                if random.random() < 0.7:
                    # Pick a violating K4 or K5 if any exist
                    violating_k4s = np.where(red_count_k4 >= 3)[0]
                    violating_k5s = np.where(blue_count_k5 >= 9)[0]
                    
                    candidates = []
                    if len(violating_k4s) > 0:
                        k4_idx = random.choice(violating_k4s)
                        candidates.extend(k4_triples[k4_idx])
                    if len(violating_k5s) > 0:
                        k5_idx = random.choice(violating_k5s)
                        candidates.extend(k5_triples[k5_idx])
                    
                    if candidates:
                        t_idx = random.choice(candidates)
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    t_idx = random.randint(0, num_triples - 1)
                
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Calculate Delta E
                delta_red = 0
                delta_blue = 0
                
                # Update K4 counts
                for k4_idx in triple_to_k4[t_idx]:
                    if old_color == 0:  # Was Red, becoming Blue
                        old_cnt = int(red_count_k4[k4_idx])
                        new_cnt = old_cnt - 1
                        if old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        elif old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        red_count_k4[k4_idx] = new_cnt
                    else:  # Was Blue, becoming Red
                        old_cnt = int(red_count_k4[k4_idx])
                        new_cnt = old_cnt + 1
                        if old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        elif old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        red_count_k4[k4_idx] = new_cnt
                        
                # Update K5 counts
                for k5_idx in triple_to_k5[t_idx]:
                    if old_color == 1:  # Was Blue, becoming Red
                        old_cnt = int(blue_count_k5[k5_idx])
                        new_cnt = old_cnt - 1
                        if old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        elif old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        blue_count_k5[k5_idx] = new_cnt
                    else:  # Was Red, becoming Blue
                        old_cnt = int(blue_count_k5[k5_idx])
                        new_cnt = old_cnt + 1
                        if old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        elif old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        blue_count_k5[k5_idx] = new_cnt
                        
                delta_E = delta_red + delta_blue
                
                accepted = False
                if delta_E < 0:
                    accepted = True
                elif T > min_T:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        accepted = True
                
                if accepted:
                    colors[t_idx] = new_color
                    red_violations += delta_red
                    blue_violations += delta_blue
                else:
                    # Rollback
                    for k4_idx in triple_to_k4[t_idx]:
                        if old_color == 0:
                            red_count_k4[k4_idx] += 1
                        else:
                            red_count_k4[k4_idx] -= 1
                    for k5_idx in triple_to_k5[t_idx]:
                        if old_color == 1:
                            blue_count_k5[k5_idx] += 1
                        else:
                            blue_count_k5[k5_idx] -= 1
                
                if red_violations == 0 and blue_violations == 0:
                    return colors
                    
                T *= cooling
                
                total_violations = red_violations + blue_violations
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_result = colors[:]
        
        return best_result
    
    # Try increasing n from high to low, allocating more time to larger n
    # Start from a reasonable n and try to go higher
    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # First, quickly verify n=9 works
    # Then try to increase
    
    # Allocate time: try n from 34 down to 10, with decreasing time budgets
    # But we want to maximize n, so try high n first with limited time, then lower n with more time
    
    # Strategy: try n=34, 33, ..., 10 with time budgets that sum to ~130s
    # Give more time to intermediate n where solutions are likely
    
    n_to_try = list(range(34, 9, -1))
    total_time = 130.0
    # Allocate time: larger n gets less time initially, but we can adjust
    # Use a schedule: n=34 gets 5s, n=33 gets 5s, ..., n=10 gets more
    time_per_n = {}
    remaining = total_time
    for i, n in enumerate(n_to_try):
        # Decreasing time as n decreases (since smaller n is easier)
        t = max(2.0, remaining * 0.15)
        time_per_n[n] = t
        remaining -= t
    
    for n in n_to_try:
        if time_left() < 3.0:
            break
        
        budget = min(time_per_n.get(n, 5.0), time_left() - 3.0)
        if budget < 1.0:
            break
        
        result = solve_n(n, time_budget=budget)
        if result is not None:
            best_n = n
            best_colors = result
            # If we found a valid coloring, continue trying higher n? 
            # No, we're going from high to low, so if we found n, we already tried higher.
            # Actually we go 34, 33, ... so if we find n=30, we've already tried 34,33,32,31.
            # We should stop if we find a solution? No, we want the MAXIMUM n.
            # Since we go from high to low, the first solution we find is the maximum.
            # So we can return immediately.
            break
    
    return {'n': best_n, 'colors': best_colors}
