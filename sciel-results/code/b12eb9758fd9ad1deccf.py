import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    
    best_n = n_fallback
    best_colors = colors_fallback
    
    targets = [34, 35, 25, 20, 15]
    
    for n in targets:
        if time.time() > deadline - 10.0:
            break
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for k4 in combinations(range(n), 4):
            k4s.append(tuple(triple_to_idx[(a,b,c)] for a,b,c in combinations(k4, 3)))
            
        k5s = []
        for k5 in combinations(range(n), 5):
            k5s.append(tuple(triple_to_idx[(a,b,c)] for a,b,c in combinations(k5, 3)))
            
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        triple_to_k4 = [[] for _ in range(num_triples)]
        for k4_idx, k4_triples in enumerate(k4s):
            for t_idx in k4_triples:
                triple_to_k4[t_idx].append(k4_idx)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for k5_idx, k5_triples in enumerate(k5s):
            for t_idx in k5_triples:
                triple_to_k5[t_idx].append(k5_idx)
                
        colors = [random.getrandbits(1) for _ in range(num_triples)]
        
        is_red_k4 = [False] * num_k4
        current_violations = 0
        for k4_idx, k4_triples in enumerate(k4s):
            if all(colors[t] == 0 for t in k4_triples):
                is_red_k4[k4_idx] = True
                current_violations += 1
                
        is_blue_k5 = [False] * num_k5
        for k5_idx, k5_triples in enumerate(k5s):
            if all(colors[t] == 1 for t in k5_triples):
                is_blue_k5[k5_idx] = True
                current_violations += 1
        
        best_violations = current_violations
        best_colors = colors.copy()
        
        T = 1.0
        T_min = 0.001
        cooling = 0.99999
        
        while time.time() < deadline - 5.0:
            if current_violations == 0:
                best_violations = 0
                best_colors = colors.copy()
                break
                
            t_idx = random.getrandbits(30) % num_triples
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta = 0
            
            for k4_idx in triple_to_k4[t_idx]:
                if is_red_k4[k4_idx]:
                    is_red_k4[k4_idx] = False
                    delta -= 1
                else:
                    if new_color == 0:
                        k4_triples = k4s[k4_idx]
                        is_all_red = True
                        for ot in k4_triples:
                            if ot != t_idx and colors[ot] != 0:
                                is_all_red = False
                                break
                        if is_all_red:
                            is_red_k4[k4_idx] = True
                            delta += 1
                            
            for k5_idx in triple_to_k5[t_idx]:
                if is_blue_k5[k5_idx]:
                    is_blue_k5[k5_idx] = False
                    delta -= 1
                else:
                    if new_color == 1:
                        k5_triples = k5s[k5_idx]
                        is_all_blue = True
                        for ot in k5_triples:
                            if ot != t_idx and colors[ot] != 1:
                                is_all_blue = False
                                break
                        if is_all_blue:
                            is_blue_k5[k5_idx] = True
                            delta += 1
                            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[t_idx] = new_color
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        if best_violations == 0:
            best_n = n
            best_colors = best_colors
        else:
            # If we failed to find a valid solution for this n, 
            # and n is large, we might want to skip larger n's?
            # But the list is ordered 34, 35, 25...
            # If 34 fails, 35 is likely to fail.
            # We just continue to next target (35) or break if time is up.
            # To save time, if n > 20 and failed, we can break?
            # Let's just let it run, time guard is sufficient.
            pass

    return {'n': best_n, 'colors': best_colors}
