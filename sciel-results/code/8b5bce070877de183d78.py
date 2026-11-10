import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_structures(n):
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        k4s = list(itertools.combinations(range(n), 4))
        N_k4 = len(k4s)
        M4 = np.zeros((N_k4, 4), dtype=np.int32)
        triple_to_k4s = [[] for _ in range(M)]
        
        for i, k4 in enumerate(k4s):
            t1 = triple_index[(k4[0], k4[1], k4[2])]
            t2 = triple_index[(k4[0], k4[1], k4[3])]
            t3 = triple_index[(k4[0], k4[2], k4[3])]
            t4 = triple_index[(k4[1], k4[2], k4[3])]
            M4[i] = [t1, t2, t3, t4]
            triple_to_k4s[t1].append(i)
            triple_to_k4s[t2].append(i)
            triple_to_k4s[t3].append(i)
            triple_to_k4s[t4].append(i)
            
        k5s = list(itertools.combinations(range(n), 5))
        N_k5 = len(k5s)
        M5 = np.zeros((N_k5, 10), dtype=np.int32)
        triple_to_k5s = [[] for _ in range(M)]
        
        for i, k5 in enumerate(k5s):
            for j, t in enumerate(itertools.combinations(k5, 3)):
                t_idx = triple_index[t]
                M5[i, j] = t_idx
                triple_to_k5s[t_idx].append(i)
                
        return M4, M5, triple_to_k4s, triple_to_k5s

    def solve_for_n(n, time_limit):
        M4, M5, triple_to_k4s, triple_to_k5s = get_structures(n)
        M = len(triple_to_k4s)
        
        best_colors = None
        best_V = float('inf')
        
        current_time = time.time()
        
        while time.time() < current_time + time_limit:
            colors = np.random.randint(0, 2, M, dtype=np.int8)
            sums4 = colors[M4].sum(axis=1)
            sums5 = colors[M5].sum(axis=1)
            
            V = np.sum(sums4 == 0) + np.sum(sums5 == 10)
            
            if V == 0:
                return colors, M4, M5
                
            if V < best_V:
                best_V = V
                best_colors = colors.copy()
                
            T = 1.0
            T_min = 0.01
            alpha = 0.9995
            
            while time.time() < current_time + time_limit and T > T_min:
                t = np.random.randint(0, M)
                c = colors[t]
                new_c = 1 - c
                delta_sum = 1 if c == 0 else -1
                
                delta_V = 0
                
                for k4_idx in triple_to_k4s[t]:
                    old_sum = sums4[k4_idx]
                    new_sum = old_sum + delta_sum
                    old_v = 1 if old_sum == 0 else 0
                    new_v = 1 if new_sum == 0 else 0
                    delta_V += new_v - old_v
                    sums4[k4_idx] = new_sum
                    
                for k5_idx in triple_to_k5s[t]:
                    old_sum = sums5[k5_idx]
                    new_sum = old_sum + delta_sum
                    old_v = 1 if old_sum == 10 else 0
                    new_v = 1 if new_sum == 10 else 0
                    delta_V += new_v - old_v
                    sums5[k5_idx] = new_sum
                    
                if delta_V < 0 or random.random() < np.exp(-delta_V / T):
                    colors[t] = new_c
                    V += delta_V
                    if V < best_V:
                        best_V = V
                        best_colors = colors.copy()
                        if V == 0:
                            return colors, M4, M5
                else:
                    for k4_idx in triple_to_k4s[t]:
                        sums4[k4_idx] -= delta_sum
                    for k5_idx in triple_to_k5s[t]:
                        sums5[k5_idx] -= delta_sum
                            
                T *= alpha
                
            if time.time() > current_time + time_limit - 2.0:
                break
                
        return best_colors, M4, M5

    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 15:
            break
            
        time_alloc = 20.0 if n >= 30 else 10.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time() - 5.0
        if time_alloc <= 0:
            break
            
        sol, M4, M5 = solve_for_n(n, time_alloc)
        
        if sol is not None:
            sums4 = sol[M4].sum(axis=1)
            sums5 = sol[M5].sum(axis=1)
            V = np.sum(sums4 == 0) + np.sum(sums5 == 10)
            if V == 0:
                return {'n': n, 'colors': sol.tolist()}
                
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
