import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Fallback: seed coloring for n=9
    def get_seed():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    
    # Try n=34 first (known to be achievable)
    n = 34
    
    # Precomputation for n=34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build K4 list: each entry is a tuple of 4 triple indices
    k4_list = []
    for c4 in combinations(range(n), 4):
        a, b, c, d = c4
        t1 = (a, b, c)
        t2 = (a, b, d)
        t3 = (a, c, d)
        t4 = (b, c, d)
        k4_list.append((triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]))
    
    # Build K5 list: each entry is a tuple of 10 triple indices
    k5_list = []
    for c5 in combinations(range(n), 5):
        a, b, c, d, e = c5
        ts = [
            (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
            (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
        ]
        k5_list.append(tuple(triple_to_idx[t] for t in ts))
    
    # Build reverse adjacency: for each triple, which K4s and K5s contain it
    triple_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        for t_idx in k4:
            triple_k4s[t_idx].append(i)
    
    triple_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        for t_idx in k5:
            triple_k5s[t_idx].append(i)
    
    # Count initial violations
    def count_violations(colors):
        red_k4 = 0
        blue_k5 = 0
        for k4 in k4_list:
            if all(colors[i] == 0 for i in k4):
                red_k4 += 1
        for k5 in k5_list:
            if all(colors[i] == 1 for i in k5):
                blue_k5 += 1
        return red_k4 + blue_k5
    
    # Simulated Annealing with multiple restarts
    max_restarts = 10
    found = False
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        total_viol = count_violations(colors)
        
        if total_viol == 0:
            return {'n': n, 'colors': colors}
        
        # SA parameters
        T = 2.0
        cooling = 0.99995
        min_T = 0.005
        best_viol = total_viol
        best_colors = colors[:]
        
        steps = 0
        max_steps = 300000
        
        while time.time() < deadline - 3.0 and total_viol > 0 and steps < max_steps:
            # Pick a random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            # Check K4s containing triple i
            for k4_idx in triple_k4s[i]:
                k4 = k4_list[k4_idx]
                red_count_other = 0
                for j in k4:
                    if j != i and colors[j] == 0:
                        red_count_other += 1
                
                if old_color == 0 and red_count_other == 3:
                    delta -= 1
                if new_color == 0 and red_count_other == 3:
                    delta += 1
            
            # Check K5s containing triple i
            for k5_idx in triple_k5s[i]:
                k5 = k5_list[k5_idx]
                blue_count_other = 0
                for j in k5:
                    if j != i and colors[j] == 1:
                        blue_count_other += 1
                
                if old_color == 1 and blue_count_other == 9:
                    delta -= 1
                if new_color == 1 and blue_count_other == 9:
                    delta += 1
            
            # Acceptance decision
            if delta < 0:
                colors[i] = new_color
                total_viol += delta
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
            elif delta == 0:
                # Always accept neutral moves
                colors[i] = new_color
            else:
                # Accept with probability exp(-delta / T)
                try:
                    p = math.exp(-delta / T)
                except OverflowError:
                    p = 0.0
                
                if random.random() < p:
                    colors[i] = new_color
                    total_viol += delta
            
            T *= cooling
            if T < min_T:
                T = min_T
            steps += 1
        
        # If we found a valid coloring during this restart
        if best_viol == 0:
            return {'n': n, 'colors': best_colors}
        
        # If we got close, use best_colors as starting point for next restart
        if best_viol < 50:
            colors = best_colors[:]
            total_viol = best_viol
    
    # If n=34 failed, try n=33
    if time.time() < deadline - 10.0:
        n = 33
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = []
        for c4 in combinations(range(n), 4):
            a, b, c, d = c4
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            k4_list.append((triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]))
        
        k5_list = []
        for c5 in combinations(range(n), 5):
            a, b, c, d, e = c5
            ts = [
                (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
                (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
            ]
            k5_list.append(tuple(triple_to_idx[t] for t in ts))
        
        triple_k4s = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(k4_list):
            for t_idx in k4:
                triple_k4s[t_idx].append(i)
        
        triple_k5s = [[] for _ in range(num_triples)]
        for i, k5 in enumerate(k5_list):
            for t_idx in k5:
                triple_k5s[t_idx].append(i)
        
        # Quick SA for n=33
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        total_viol = count_violations(colors)
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        steps = 0
        
        while time.time() < deadline - 3.0 and total_viol > 0 and steps < 200000:
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            for k4_idx in triple_k4s[i]:
                k4 = k4_list[k4_idx]
                red_count_other = 0
                for j in k4:
                    if j != i and colors[j] == 0:
                        red_count_other += 1
                if old_color == 0 and red_count_other == 3:
                    delta -= 1
                if new_color == 0 and red_count_other == 3:
                    delta += 1
            
            for k5_idx in triple_k5s[i]:
                k5 = k5_list[k5_idx]
                blue_count_other = 0
                for j in k5:
                    if j != i and colors[j] == 1:
                        blue_count_other += 1
                if old_color == 1 and blue_count_other == 9:
                    delta -= 1
                if new_color == 1 and blue_count_other == 9:
                    delta += 1
            
            if delta < 0:
                colors[i] = new_color
                total_viol += delta
            elif delta == 0:
                colors[i] = new_color
            else:
                try:
                    p = math.exp(-delta / T)
                except OverflowError:
                    p = 0.0
                if random.random() < p:
                    colors[i] = new_color
                    total_viol += delta
            
            T *= cooling
            if T < min_T:
                T = min_T
            steps += 1
        
        if total_viol == 0:
            return {'n': n, 'colors': colors}
    
    # Fallback to seed
    return get_seed()
