import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
        if n == 34:
            budget = min(80.0, deadline - time.time())
        elif n == 33:
            budget = min(15.0, deadline - time.time())
        elif n == 32:
            budget = min(5.0, deadline - time.time())
        elif n == 31:
            budget = min(2.0, deadline - time.time())
        elif n == 30:
            budget = min(1.0, deadline - time.time())
        else:
            budget = min(0.5, deadline - time.time())
        if budget < 0.3:
            continue
            
        try:
            result = solve_n(n, budget)
            if result is not None:
                return result
        except Exception:
            continue
            
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_n(n, time_limit):
    start = time.time()
    
    triples = list(combinations(range(n), 3))
    M = len(triples)
    idx = {t: i for i, t in enumerate(triples)}
    
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    triple_to_4 = [[] for _ in range(M)]
    for i4, s4 in enumerate(sets4):
        for j in range(4):
            t = tuple(sorted(s4[:j] + s4[j+1:]))
            triple_to_4[idx[t]].append(i4)
            
    triple_to_5 = [[] for _ in range(M)]
    for i5, s5 in enumerate(sets5):
        for j in range(5):
            t = tuple(sorted(s5[:j] + s5[j+1:]))
            triple_to_5[idx[t]].append(i5)
    
    s4_triples = []
    for s4 in sets4:
        ts = [idx[tuple(sorted(s4[:j] + s4[j+1:]))] for j in range(4)]
        s4_triples.append(ts)
        
    s5_triples = []
    for s5 in sets5:
        ts = [idx[t] for t in combinations(s5, 3)]
        s5_triples.append(ts)
    
    max_restarts = 500
    best_viol = float('inf')
    best_colors = None
    
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            break
            
        colors = np.random.randint(0, 2, M)
        
        red_count_4 = np.zeros(N4, dtype=np.int16)
        blue_count_5 = np.zeros(N5, dtype=np.int16)
        
        for i in range(M):
            c = colors[i]
            for i4 in triple_to_4[i]:
                if c == 0:
                    red_count_4[i4] += 1
            for i5 in triple_to_5[i]:
                if c == 1:
                    blue_count_5[i5] += 1
                    
        viol_4 = np.sum(red_count_4 == 4)
        viol_5 = np.sum(blue_count_5 == 10)
        total_viol = viol_4 + viol_5
        
        if total_viol == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        T = 1.0
        max_iters = 500
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
                
            if total_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            t_idx = random.randint(0, M - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta = 0
            if old_color == 0:
                for i4 in triple_to_4[t_idx]:
                    if red_count_4[i4] == 4:
                        delta -= 1
                for i5 in triple_to_5[t_idx]:
                    if blue_count_5[i5] == 9:
                        delta += 1
                    elif blue_count_5[i5] == 10:
                        delta -= 1
            else:
                for i4 in triple_to_4[t_idx]:
                    if red_count_4[i4] == 3:
                        delta += 1
                    elif red_count_4[i4] == 4:
                        delta -= 1
                for i5 in triple_to_5[t_idx]:
                    if blue_count_5[i5] == 10:
                        delta -= 1
                    elif blue_count_5[i5] == 9:
                        delta += 1
                        
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[t_idx] = new_color
                total_viol += delta
                
                for i4 in triple_to_4[t_idx]:
                    if old_color == 0:
                        red_count_4[i4] -= 1
                    else:
                        red_count_4[i4] += 1
                        
                for i5 in triple_to_5[t_idx]:
                    if old_color == 1:
                        blue_count_5[i5] -= 1
                    else:
                        blue_count_5[i5] += 1
                        
            T *= 0.995
            
        if total_viol < best_viol:
            best_viol = total_viol
            best_colors = colors.copy()
            
    if best_colors is not None and best_viol == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    return None
