import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    targets = [34, 33, 32, 30, 28, 25, 20, 15, 10]
    
    for n in targets:
        if time.time() > deadline:
            break
            
        # Allocate time
        # 34: 60s, 33: 40s, 32: 20s, others: 10s
        if n >= 34:
            time_budget = 60.0
        elif n >= 32:
            time_budget = 40.0
        elif n >= 30:
            time_budget = 20.0
        else:
            time_budget = 10.0
            
        result = try_find_coloring(n, time_budget, deadline)
        if result is not None:
            return result
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def try_find_coloring(n, time_budget, deadline):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    triples_in_k4 = [[] for _ in range(num_triples)]
    triples_in_k5 = [[] for _ in range(num_triples)]
    
    k4_triples = []
    for j, k4 in enumerate(k4s):
        t_indices = []
        for t in combinations(k4, 3):
            t_indices.append(triple_to_idx[t])
        k4_triples.append(t_indices)
        for ti in t_indices:
            triples_in_k4[ti].append(j)
            
    k5_triples = []
    for k, k5 in enumerate(k5s):
        t_indices = []
        for t in combinations(k5, 3):
            t_indices.append(triple_to_idx[t])
        k5_triples.append(t_indices)
        for ti in t_indices:
            triples_in_k5[ti].append(k)
            
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    max_iters = 1000000 # Rough estimate
    
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    k4_red_counts = [0] * num_k4
    k5_blue_counts = [0] * num_k5
    
    for i, col in enumerate(colors):
        for j in triples_in_k4[i]:
            if col == 0:
                k4_red_counts[j] += 1
        for k in triples_in_k5[i]:
            if col == 1:
                k5_blue_counts[k] += 1
                
    violations = sum(1 for c in k4_red_counts if c == 4) + sum(1 for c in k5_blue_counts if c == 10)
    
    best_violations = violations
    best_colors = colors[:]
    
    current_time = time.time()
    total_time = time_budget
    
    for step in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_iters)
        
        i = random.randint(0, num_triples - 1)
        col = colors[i]
        
        delta = 0
        for j in triples_in_k4[i]:
            if col == 0:
                if k4_red_counts[j] == 4:
                    delta -= 1
            else:
                if k4_red_counts[j] == 3:
                    delta += 1
        for k in triples_in_k5[i]:
            if col == 1:
                if k5_blue_counts[k] == 10:
                    delta -= 1
            else:
                if k5_blue_counts[k] == 9:
                    delta += 1
                    
        if delta < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta / T)
            else:
                accept = False
                
        if accept:
            colors[i] = 1 - col
            violations += delta
            
            for j in triples_in_k4[i]:
                if col == 0:
                    k4_red_counts[j] -= 1
                else:
                    k4_red_counts[j] += 1
            for k in triples_in_k5[i]:
                if col == 1:
                    k5_blue_counts[k] -= 1
                else:
                    k5_blue_counts[k] += 1
                    
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
    if violations == 0:
        return {'n': n, 'colors': colors}
    elif best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
