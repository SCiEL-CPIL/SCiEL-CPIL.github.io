import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_indices(n):
        triples = get_triples(n)
        t_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        k5s = []
        
        for combo in combinations(range(n), 4):
            ks = [t_idx[t] for t in combinations(combo, 3)]
            k4s.append(ks)
            
        for combo in combinations(range(n), 5):
            ks = [t_idx[t] for t in combinations(combo, 3)]
            k5s.append(ks)
            
        # Map triple index to list of k4 indices and k5 indices
        k4_by_t = [[] for _ in triples]
        for i, ks in enumerate(k4s):
            for k in ks:
                k4_by_t[k].append(i)
                
        k5_by_t = [[] for _ in triples]
        for i, ks in enumerate(k5s):
            for k in ks:
                k5_by_t[k].append(i)
                
        return triples, t_idx, k4s, k5s, k4_by_t, k5_by_t

    def try_n(n, time_limit):
        triples, t_idx, k4s, k5s, k4_by_t, k5_by_t = build_indices(n)
        num_triples = len(triples)
        
        # Initialize colors randomly
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_count_k4 = [0] * len(k4s)
        blue_count_k5 = [0] * len(k5s)
        
        for i, ks in enumerate(k4s):
            for k in ks:
                if colors[k] == 0:
                    red_count_k4[i] += 1
                    
        for i, ks in enumerate(k5s):
            for k in ks:
                if colors[k] == 1:
                    blue_count_k5[i] += 1
        
        current_cost = sum(1 for c in red_count_k4 if c == 4) + sum(1 for c in blue_count_k5 if c == 10)
        best_cost = current_cost
        best_colors = colors.copy()
        best_red_count = red_count_k4.copy()
        best_blue_count = blue_count_k5.copy()
        
        temp = 1.0
        cooling = 0.999
        local_deadline = time.time() + time_limit

        while time.time() < local_deadline:
            t_idx_rand = random.randint(0, num_triples - 1)
            
            delta = 0
            cur_color = colors[t_idx_rand]
            
            if cur_color == 0: # Red -> Blue
                # Update K4s
                for k4_i in k4_by_t[t_idx_rand]:
                    if red_count_k4[k4_i] == 4:
                        delta -= 1
                    red_count_k4[k4_i] -= 1
                    
                # Update K5s
                for k5_i in k5_by_t[t_idx_rand]:
                    if blue_count_k5[k5_i] == 10:
                        delta -= 1
                    blue_count_k5[k5_i] -= 1
                    
                colors[t_idx_rand] = 1
                
            else: # Blue -> Red
                # Update K4s
                for k4_i in k4_by_t[t_idx_rand]:
                    if red_count_k4[k4_i] == 3:
                        delta += 1
                    red_count_k4[k4_i] += 1
                    
                # Update K5s
                for k5_i in k5_by_t[t_idx_rand]:
                    if blue_count_k5[k5_i] == 9:
                        delta += 1
                    blue_count_k5[k5_i] += 1
                    
                colors[t_idx_rand] = 0
            
            new_cost = current_cost + delta
            
            if new_cost < current_cost or (new_cost == current_cost and random.random() < 0.5):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    best_red_count = red_count_k4.copy()
                    best_blue_count = blue_count_k5.copy()
                    if best_cost == 0:
                        return best_colors
            else:
                # Revert
                if cur_color == 0: # Was Red, now Blue, revert to Red
                    for k4_i in k4_by_t[t_idx_rand]:
                        red_count_k4[k4_i] += 1
                    for k5_i in k5_by_t[t_idx_rand]:
                        blue_count_k5[k5_i] += 1
                    colors[t_idx_rand] = 0
                else: # Was Blue, now Red, revert to Blue
                    for k4_i in k4_by_t[t_idx_rand]:
                        red_count_k4[k4_i] -= 1
                    for k5_i in k5_by_t[t_idx_rand]:
                        blue_count_k5[k5_i] -= 1
                    colors[t_idx_rand] = 1
                
                if new_cost > current_cost:
                    if random.random() < pow(2.71828, -delta / temp):
                        # Accept bad move
                        current_cost = new_cost
                        # Re-apply the move
                        if cur_color == 0:
                            for k4_i in k4_by_t[t_idx_rand]:
                                red_count_k4[k4_i] -= 1
                            for k5_i in k5_by_t[t_idx_rand]:
                                blue_count_k5[k5_i] -= 1
                            colors[t_idx_rand] = 1
                        else:
                            for k4_i in k4_by_t[t_idx_rand]:
                                red_count_k4[k4_i] += 1
                            for k5_i in k5_by_t[t_idx_rand]:
                                blue_count_k5[k5_i] += 1
                            colors[t_idx_rand] = 0
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01

        if best_cost == 0:
            return best_colors
        return None

    best_result = None
    targets = [(15, 30), (14, 30), (13, 20), (12, 20)]
    
    for n, t_lim in targets:
        if time.time() > deadline - 5:
            break
        colors = try_n(n, t_lim)
        if colors is not None:
            best_result = {'n': n, 'colors': colors}
            break
            
    if best_result is None:
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_result = {'n': n, 'colors': colors}
        
    return best_result
