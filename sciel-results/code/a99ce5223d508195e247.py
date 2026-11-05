import itertools
import random
import time
import numpy as np

def construct():
    max_n = 34
    start_time = time.time()
    time_limit = 110
    
    # Precompute structures for n=34 (and potentially smaller if we fallback, but we aim for 34)
    # We will try n=34 first.
    
    for n in range(max_n, 10, -1):
        if time.time() - start_time > time_limit:
            break
            
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        # Map each triple to the 4-sets and 5-sets it belongs to
        # triple_idx -> list of 4-set indices
        # triple_idx -> list of 5-set indices
        t_to_4 = [[] for _ in range(num_triples)]
        t_to_5 = [[] for _ in range(num_triples)]
        
        # 4-set indices for each 4-set
        # 4-set_idx -> list of 3 triple indices
        set4_triples = []
        for idx, s in enumerate(four_sets):
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            set4_triples.append(ts)
            for t_idx in ts:
                t_to_4[t_idx].append(idx)
                
        # 5-set indices for each 5-set
        # 5-set_idx -> list of 10 triple indices
        set5_triples = []
        for idx, s in enumerate(five_sets):
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            set5_triples.append(ts)
            for t_idx in ts:
                t_to_5[t_idx].append(idx)

        # Try multiple random restarts
        found = False
        # Number of restarts depends on time, but let's say 50 restarts
        # Each restart runs SA for a fixed number of steps or time slice
        
        for seed in range(100):
            if time.time() - start_time > time_limit:
                break
                
            random.seed(seed)
            # Initialize colors randomly
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Initialize counts
            red_count_4 = [0] * len(four_sets)
            blue_count_5 = [0] * len(five_sets)
            
            for t_idx, c in enumerate(colors):
                for s_idx in t_to_4[t_idx]:
                    if c == 0:
                        red_count_4[s_idx] += 1
                for s_idx in t_to_5[t_idx]:
                    if c == 1:
                        blue_count_5[s_idx] += 1
                        
            # Calculate penalty
            penalty = sum(1 for r in red_count_4 if r == 4) + sum(1 for b in blue_count_5 if b == 10)
            
            if penalty == 0:
                return {'n': n, 'colors': colors}
                
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            max_iters = 20000
            current_iter = 0
            
            while current_iter < max_iters:
                if time.time() - start_time > time_limit:
                    break
                    
                if penalty == 0:
                    found = True
                    break
                    
                # Pick a random triple to flip
                t_idx = random.randint(0, num_triples - 1)
                c = colors[t_idx]
                new_c = 1 - c
                
                # Calculate delta penalty
                delta = 0
                
                # Update red counts for 4-sets
                for s_idx in t_to_4[t_idx]:
                    if c == 0: # Red -> Blue
                        # red count decreases
                        if red_count_4[s_idx] == 4:
                            delta -= 1 # Fixed a red K4
                        # If it was 3, becomes 2, no penalty change
                    else: # Blue -> Red
                        # red count increases
                        if red_count_4[s_idx] == 3:
                            delta += 1 # Created a red K4
                        
                # Update blue counts for 5-sets
                for s_idx in t_to_5[t_idx]:
                    if c == 1: # Blue -> Red
                        # blue count decreases
                        if blue_count_5[s_idx] == 10:
                            delta -= 1 # Fixed a blue K5
                    else: # Red -> Blue
                        # blue count increases
                        if blue_count_5[s_idx] == 9:
                            delta += 1 # Created a blue K5
                            
                # Acceptance probability
                if delta < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta/T)
                    if delta == 0:
                        accept = random.random() < 0.5
                    else:
                        accept = random.random() < np.exp(-delta / T)
                        
                if accept:
                    # Apply flip
                    colors[t_idx] = new_c
                    penalty += delta
                    
                    # Update counts
                    for s_idx in t_to_4[t_idx]:
                        if c == 0: # Was Red, now Blue
                            red_count_4[s_idx] -= 1
                        else:
                            red_count_4[s_idx] += 1
                            
                    for s_idx in t_to_5[t_idx]:
                        if c == 1: # Was Blue, now Red
                            blue_count_5[s_idx] -= 1
                        else:
                            blue_count_5[s_idx] += 1
                            
                    T *= cooling
                    if T < T_min:
                        T = T_min
                else:
                    # Optional: don't cool down on reject? Or cool down slowly?
                    # Standard SA cools regardless.
                    T *= cooling
                    if T < T_min:
                        T = T_min
                        
                current_iter += 1
                
            if found:
                return {'n': n, 'colors': colors}
                
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
