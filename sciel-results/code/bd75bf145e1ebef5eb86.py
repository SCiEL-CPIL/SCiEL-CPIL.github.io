import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # We will try to find a valid coloring for n=34.
    # If successful, return it.
    # If not, try n=33, etc.
    
    # Known bound is 34. We aim for 34.
    
    def find_coloring(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        # Map triple index to list of K4 indices containing it
        # Map triple index to list of K5 indices containing it
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        # We also need to know which triples are in which K4/K5 to update counts?
        # No, we only need to update counts for the K4s/K5s that contain the flipped triple.
        # So we just need the reverse mapping: triple -> [k4_indices], triple -> [k5_indices].
        
        for i, k4 in enumerate(k4_list):
            # 4 triples in a K4
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                triple_to_k4s[idx].append(i)
                
        for i, k5 in enumerate(k5_list):
            # 10 triples in a K5
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                triple_to_k5s[idx].append(i)
                
        # Precompute lengths for faster access? No, lists are fine.
        
        def run_sa(n, triples, num_triples, triple_to_k4s, triple_to_k5s, num_k4, num_k5, time_limit):
            # Initial random coloring
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
            
            # Initialize counts
            # R[i] = number of red triples in K4 i
            # B[i] = number of blue triples in K5 i
            R = np.zeros(num_k4, dtype=np.int16)
            B = np.zeros(num_k5, dtype=np.int16)
            
            for t_idx in range(num_triples):
                if colors[t_idx] == 0: # Red
                    for k4_idx in triple_to_k4s[t_idx]:
                        R[k4_idx] += 1
                else: # Blue
                    for k5_idx in triple_to_k5s[t_idx]:
                        B[k5_idx] += 1
                        
            # Count violations
            red_violations = np.sum(R == 4)
            blue_violations = np.sum(B == 10)
            total_violations = red_violations + blue_violations
            
            if total_violations == 0:
                return colors.copy()
            
            best_violations = total_violations
            best_colors = colors.copy()
            
            T = 50.0
            T_end = 0.01
            cooling_rate = 0.9999
            last_improvement_time = time.time()
            
            while time.time() < time_limit:
                if total_violations == 0:
                    return colors.copy()
                    
                # Pick random triple
                t_idx = random.randint(0, num_triples - 1)
                old_color = colors[t_idx]
                
                # Calculate Delta
                delta = 0
                
                if old_color == 0: # Red -> Blue
                    # R decreases for K4s containing t
                    # B increases for K5s containing t
                    
                    # Count K4s with R=4 (will become 3, removing violation)
                    # Note: If R=4, t must be red. Since t is red, this is valid.
                    for k4_idx in triple_to_k4s[t_idx]:
                        if R[k4_idx] == 4:
                            delta -= 1
                            
                    # Count K5s with B=9 (will become 10, creating violation)
                    # Note: If B=9, t is not blue (since we are adding blue). t is red. Valid.
                    for k5_idx in triple_to_k5s[t_idx]:
                        if B[k5_idx] == 9:
                            delta += 1
                            
                else: # Blue -> Red
                    # R increases for K4s containing t
                    # B decreases for K5s containing t
                    
                    # Count K4s with R=3 (will become 4, creating violation)
                    for k4_idx in triple_to_k4s[t_idx]:
                        if R[k4_idx] == 3:
                            delta += 1
                            
                    # Count K5s with B=10 (will become 9, removing violation)
                    for k5_idx in triple_to_k5s[t_idx]:
                        if B[k5_idx] == 10:
                            delta -= 1
                            
                # Acceptance
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = True
                else:
                    if T > T_end:
                        prob = np.exp(-delta / T)
                        accept = random.random() < prob
                    else:
                        accept = False
                        
                if accept:
                    colors[t_idx] = 1 - old_color
                    total_violations += delta
                    
                    # Update counts
                    if old_color == 0: # Was Red, now Blue
                        for k4_idx in triple_to_k4s[t_idx]:
                            R[k4_idx] -= 1
                        for k5_idx in triple_to_k5s[t_idx]:
                            B[k5_idx] += 1
                    else: # Was Blue, now Red
                        for k4_idx in triple_to_k4s[t_idx]:
                            R[k4_idx] += 1
                        for k5_idx in triple_to_k5s[t_idx]:
                            B[k5_idx] -= 1
                            
                    if total_violations < best_violations:
                        best_violations = total_violations
                        best_colors = colors.copy()
                        last_improvement_time = time.time()
                        
                T *= cooling_rate
                
                # Restart if stuck
                if time.time() - last_improvement_time > 2.0:
                    colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                    R = np.zeros(num_k4, dtype=np.int16)
                    B = np.zeros(num_k5, dtype=np.int16)
                    for t_idx in range(num_triples):
                        if colors[t_idx] == 0:
                            for k4_idx in triple_to_k4s[t_idx]:
                                R[k4_idx] += 1
                        else:
                            for k5_idx in triple_to_k5s[t_idx]:
                                B[k5_idx] += 1
                    total_violations = np.sum(R == 4) + np.sum(B == 10)
                    T = 50.0
                    last_improvement_time = time.time()
                    
                    if total_violations < best_violations:
                        best_violations = total_violations
                        best_colors = colors.copy()
                        
            if best_violations == 0:
                return best_colors
            return None

    # Try n=34
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    num_k4 = len(k4_list)
    num_k5 = len(k5_list)
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for i, k4 in enumerate(k4_list):
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            triple_to_k4s[idx].append(i)
            
    for i, k5 in enumerate(k5_list):
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            triple_to_k5s[idx].append(i)
            
    # Allocate most time to n=34
    time_left = deadline - time.time()
    if time_left > 10:
        result = run_sa(34, triples, num_triples, triple_to_k4s, triple_to_k5s, num_k4, num_k5, deadline - 5)
        if result is not None:
            return {'n': 34, 'colors': result.tolist()}
            
    # If n=34 failed, try n=33
    if time.time() < deadline - 5:
        n = 33
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4_list):
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                idx = triple_to_idx[t]
                triple_to_k4s[idx].append(i)
                
        for i, k5 in enumerate(k5_list):
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[j] for j in combo]))
                idx = triple_to_idx[t]
                triple_to_k5s[idx].append(i)
                
        result = run_sa(33, triples, num_triples, triple_to_k4s, triple_to_k5s, num_k4, num_k5, deadline - 2)
        if result is not None:
            return {'n': 33, 'colors': result.tolist()}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
