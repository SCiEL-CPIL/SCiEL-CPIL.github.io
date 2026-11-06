import numpy as np
import time
import itertools
import random

def construct():
    deadline = time.time() + 115  # 5s safety margin for 120s limit
    
    def try_solve(n, time_budget):
        if n < 1:
            return None
        
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        sets_4 = list(itertools.combinations(range(n), 4))
        sets_5 = list(itertools.combinations(range(n), 5))
        
        num_sets_4 = len(sets_4)
        num_sets_5 = len(sets_5)
        
        triples_in_4 = []
        for s in sets_4:
            idxs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            triples_in_4.append(idxs)
            
        triples_in_5 = []
        for s in sets_5:
            idxs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            triples_in_5.append(idxs)
            
        sets_4_containing = [[] for _ in range(num_triples)]
        sets_5_containing = [[] for _ in range(num_triples)]
        
        for i_4, idxs in enumerate(triples_in_4):
            for t_idx in idxs:
                sets_4_containing[t_idx].append(i_4)
                
        for i_5, idxs in enumerate(triples_in_5):
            for t_idx in idxs:
                sets_5_containing[t_idx].append(i_5)

        # Random initialization
        colors = np.random.randint(0, 2, num_triples)
        
        red_k4_count = np.zeros(num_sets_4, dtype=np.int8)
        blue_k5_count = np.zeros(num_sets_5, dtype=np.int8)
        
        for i_4, idxs in enumerate(triples_in_4):
            red_k4_count[i_4] = np.sum(colors[idxs] == 0)
            
        for i_5, idxs in enumerate(triples_in_5):
            blue_k5_count[i_5] = np.sum(colors[idxs] == 1)
            
        current_violations = np.sum(red_k4_count == 4) + np.sum(blue_k5_count == 10)
        
        if current_violations == 0:
            return colors
            
        T = 1.0
        cooling = 0.99995
        min_T = 0.001
        
        while time.time() < time_budget:
            if current_violations == 0:
                return colors
                
            t_idx = np.random.randint(num_triples)
            c_old = colors[t_idx]
            c_new = 1 - c_old
            
            delta_violations = 0
            
            for i_4 in sets_4_containing[t_idx]:
                old_red = red_k4_count[i_4]
                if c_old == 0:
                    new_red = old_red - 1
                else:
                    new_red = old_red + 1
                v_old = 1 if old_red == 4 else 0
                v_new = 1 if new_red == 4 else 0
                delta_violations += (v_new - v_old)
                
            for i_5 in sets_5_containing[t_idx]:
                old_blue = blue_k5_count[i_5]
                if c_old == 1:
                    new_blue = old_blue - 1
                else:
                    new_blue = old_blue + 1
                v_old = 1 if old_blue == 10 else 0
                v_new = 1 if new_blue == 10 else 0
                delta_violations += (v_new - v_old)
                
            exp_arg = -delta_violations / T
            if exp_arg > 100:
                prob = 1.0
            elif exp_arg < -100:
                prob = 0.0
            else:
                prob = np.exp(exp_arg)
                
            if delta_violations < 0 or (T > min_T and np.random.random() < prob):
                colors[t_idx] = c_new
                
                for i_4 in sets_4_containing[t_idx]:
                    if c_old == 0:
                        red_k4_count[i_4] -= 1
                    else:
                        red_k4_count[i_4] += 1
                        
                for i_5 in sets_5_containing[t_idx]:
                    if c_old == 1:
                        blue_k5_count[i_5] -= 1
                    else:
                        blue_k5_count[i_5] += 1
                        
                current_violations += delta_violations
                T *= cooling
                
        return None

    # Try 35
    t_start = time.time()
    colors_35 = try_solve(35, time.time() + 50)
    if colors_35 is not None:
        return {'n': 35, 'colors': colors_35.tolist()}
        
    # Try 34
    t_start = time.time()
    colors_34 = try_solve(34, time.time() + 50)
    if colors_34 is not None:
        return {'n': 34, 'colors': colors_34.tolist()}
        
    # Fallback
    n = 30
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
