import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def get_structures(n):
        triples = list(combinations(range(n), 3))
        triple_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        k4s = list(combinations(range(n), 4))
        k4_triples = []
        for k4 in k4s:
            ts = [triple_idx[(k4[i], k4[j], k4[k])] 
                  for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
            k4_triples.append(ts)
        num_k4 = len(k4_triples)
        
        k5s = list(combinations(range(n), 5))
        k5_triples = []
        for k5 in k5s:
            ts = [triple_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(ts)
        num_k5 = len(k5_triples)
        
        # triple_to_k4: for each triple, which K4s contain it
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        return triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5
    
    def count_violations(colors, red_k4_count, blue_k5_count):
        # Count number of bad K4s (all 4 red) and bad K5s (all 10 blue)
        bad_k4 = sum(1 for c in red_k4_count if c == 4)
        bad_k5 = sum(1 for c in blue_k5_count if c == 10)
        return bad_k4 + bad_k5
    
    def try_solve(n, time_limit, init_method='random'):
        if n < 4:
            return None
        
        triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5 = get_structures(n)
        num_triples = len(triples)
        num_k4 = len(k4_triples)
        num_k5 = len(k5_triples)
        
        # Initialize colors
        if init_method == 'random':
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif init_method == 'cyclic':
            # Cyclic construction: color based on sum mod some value
            colors = np.zeros(num_triples, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                s = (a + b + c) % 5
                colors[i] = 0 if s in (0, 1) else 1  # ~40% red, 60% blue
        elif init_method == 'mod3':
            colors = np.zeros(num_triples, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        elif init_method == 'mod4':
            colors = np.zeros(num_triples, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % 4 < 2 else 1
        else:
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Initialize counts
        red_k4_count = np.zeros(num_k4, dtype=np.int32)
        blue_k5_count = np.zeros(num_k5, dtype=np.int32)
        
        for i, ts in enumerate(k4_triples):
            red_k4_count[i] = sum(1 for t in ts if colors[t] == 0)
            
        for i, ts in enumerate(k5_triples):
            blue_k5_count[i] = sum(1 for t in ts if colors[t] == 1)
        
        total_viol = count_violations(colors, red_k4_count, blue_k5_count)
        if total_viol == 0:
            return colors.tolist()
        
        best_colors = colors.copy()
        best_viol = total_viol
        current_temp = 1.0
        cooling_rate = 0.999
        
        # Precompute for speed: for each triple, the list of (k4_idx, delta_red) and (k5_idx, delta_blue)
        # When we flip triple t from 0->1:
        #   For each k4 in triple_to_k4[t]: red_k4_count[k4] -= 1
        #   For each k5 in triple_to_k5[t]: blue_k5_count[k5] += 1
        # When we flip triple t from 1->0:
        #   For each k4 in triple_to_k4[t]: red_k4_count[k4] += 1
        #   For each k5 in triple_to_k5[t]: blue_k5_count[k5] -= 1
        
        steps = 0
        while time.time() < time_limit:
            # Pick a triple to flip
            # Strategy: pick from triples involved in violations, or random
            if random.random() < 0.7:
                # Pick a triple that's in a bad K4 or bad K5
                bad_k4_list = [i for i in range(num_k4) if red_k4_count[i] == 4]
                bad_k5_list = [i for i in range(num_k5) if blue_k5_count[i] == 10]
                
                candidates = set()
                if bad_k4_list:
                    k4_idx = random.choice(bad_k4_list)
                    for t in k4_triples[k4_idx]:
                        candidates.add(t)
                if bad_k5_list:
                    k5_idx = random.choice(bad_k5_list)
                    for t in k5_triples[k5_idx]:
                        candidates.add(t)
                
                if candidates:
                    t_idx = random.choice(list(candidates))
                else:
                    t_idx = random.randint(0, num_triples - 1)
            else:
                t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Compute delta in violations
            delta_viol = 0
            
            # For K4s containing t_idx
            for k4_i in triple_to_k4[t_idx]:
                old_cnt = red_k4_count[k4_i]
                if old_color == 0:  # 0->1, red count decreases
                    new_cnt = old_cnt - 1
                    if old_cnt == 4 and new_cnt != 4:
                        delta_viol -= 1  # fixed a bad K4
                    elif old_cnt != 4 and new_cnt == 4:
                        delta_viol += 1  # created a bad K4
                else:  # 1->0, red count increases
                    new_cnt = old_cnt + 1
                    if old_cnt == 4 and new_cnt != 4:
                        delta_viol -= 1
                    elif old_cnt != 4 and new_cnt == 4:
                        delta_viol += 1
            
            # For K5s containing t_idx
            for k5_i in triple_to_k5[t_idx]:
                old_cnt = blue_k5_count[k5_i]
                if old_color == 1:  # 1->0, blue count decreases
                    new_cnt = old_cnt - 1
                    if old_cnt == 10 and new_cnt != 10:
                        delta_viol -= 1  # fixed a bad K5
                    elif old_cnt != 10 and new_cnt == 10:
                        delta_viol += 1  # created a bad K5
                else:  # 0->1, blue count increases
                    new_cnt = old_cnt + 1
                    if old_cnt == 10 and new_cnt != 10:
                        delta_viol -= 1
                    elif old_cnt != 10 and new_cnt == 10:
                        delta_viol += 1
            
            new_viol = total_viol + delta_viol
            
            # Accept or reject
            if new_viol <= total_viol or random.random() < np.exp(-delta_viol / max(current_temp, 0.01)):
                # Apply flip
                colors[t_idx] = new_color
                
                for k4_i in triple_to_k4[t_idx]:
                    if old_color == 0:
                        red_k4_count[k4_i] -= 1
                    else:
                        red_k4_count[k4_i] += 1
                        
                for k5_i in triple_to_k5[t_idx]:
                    if old_color == 1:
                        blue_k5_count[k5_i] -= 1
                    else:
                        blue_k5_count[k5_i] += 1
                
                total_viol = new_viol
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    if total_viol == 0:
                        return colors.tolist()
            
            current_temp *= cooling_rate
            steps += 1
            
            # Restart if stuck
            if steps % 5000 == 0 and current_temp < 0.05:
                # Reset temperature
                current_temp = 0.5
                # Optionally perturb
                if random.random() < 0.3:
                    for _ in range(5):
                        t = random.randint(0, num_triples - 1)
                        colors[t] = 1 - colors[t]
                        for k4_i in triple_to_k4[t]:
                            if colors[t] == 0:
                                red_k4_count[k4_i] += 1
                            else:
                                red_k4_count[k4_i] -= 1
                        for k5_i in triple_to_k5[t]:
                            if colors[t] == 1:
                                blue_k5_count[k5_i] += 1
                            else:
                                blue_k5_count[k5_i] -= 1
                        total_viol = count_violations(colors, red_k4_count, blue_k5_count)
        
        # Return best found
        if best_viol == 0:
            return best_colors.tolist()
        return None
    
    # Try multiple initialization strategies and n values
    init_methods = ['cyclic', 'mod3', 'mod4', 'random', 'cyclic', 'random']
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() >= deadline - 10:
            break
            
        remaining = deadline - time.time() - 2
        if remaining < 5:
            break
            
        # Split time among init methods
        time_per_init = remaining / len(init_methods)
        
        for init in init_methods:
            if time.time() >= deadline - 5:
                break
            t_limit = time.time() + time_per_init
            colors = try_solve(n, t_limit, init_method=init)
            if colors is not None:
                return {'n': n, 'colors': colors}
    
    # Fallback: try smaller n quickly
    for n in [12, 10, 9]:
        if time.time() >= deadline - 3:
            break
        colors = try_solve(n, time.time() + 2, init_method='mod3')
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Ultimate fallback: seed solution
    n = 9
    seed_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': seed_colors}

