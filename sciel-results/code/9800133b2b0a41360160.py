"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def solve_n(n):
        if n < 4:
            return None
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
        
        # Modular warm-start initialization
        # Try a few structured initializations
        inits = []
        # mod 3
        inits.append([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples])
        # mod 2
        inits.append([0 if (a+b+c) % 2 == 0 else 1 for a,b,c in triples])
        # random
        inits.append([random.randint(0,1) for _ in range(num_triples)])
        
        best_result = None
        best_violations = float('inf')
        
        for init_colors in inits:
            if time_left() < 3.0:
                break
            
            colors = init_colors[:]
            
            # Initialize counts
            red_count_k4 = [0] * num_k4
            blue_count_k5 = [0] * num_k5
            
            for i, ts in enumerate(k4_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 0:
                        cnt += 1
                red_count_k4[i] = cnt
                
            for i, ts in enumerate(k5_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 1:
                        cnt += 1
                blue_count_k5[i] = cnt
            
            red_violations = sum(1 for c in red_count_k4 if c == 4)
            blue_violations = sum(1 for c in blue_count_k5 if c == 10)
            total_viol = red_violations + blue_violations
            
            if total_viol == 0:
                return colors
            
            if total_viol < best_violations:
                best_violations = total_viol
                best_result = colors[:]
            
            # Simulated Annealing with violation-biased sampling
            T = 1.0
            cooling = 0.9995
            min_T = 0.005
            max_steps = 500000
            no_improve = 0
            patience = 20000
            
            for step in range(max_steps):
                if time_left() < 3.0:
                    break
                
                if red_violations == 0 and blue_violations == 0:
                    return colors
                
                # Pick a triple to flip
                # Bias towards violating sets
                if red_violations > 0 and blue_violations > 0:
                    # Weighted random: pick from red violations or blue violations
                    if random.random() < red_violations / (red_violations + blue_violations):
                        # Pick a random violating K4
                        while True:
                            k4_idx = random.randint(0, num_k4 - 1)
                            if red_count_k4[k4_idx] == 4:
                                break
                        t_idx = random.choice(k4_triples[k4_idx])
                    else:
                        # Pick a random violating K5
                        while True:
                            k5_idx = random.randint(0, num_k5 - 1)
                            if blue_count_k5[k5_idx] == 10:
                                break
                        t_idx = random.choice(k5_triples[k5_idx])
                elif red_violations > 0:
                    while True:
                        k4_idx = random.randint(0, num_k4 - 1)
                        if red_count_k4[k4_idx] == 4:
                            break
                    t_idx = random.choice(k4_triples[k4_idx])
                elif blue_violations > 0:
                    while True:
                        k5_idx = random.randint(0, num_k5 - 1)
                        if blue_count_k5[k5_idx] == 10:
                            break
                    t_idx = random.choice(k5_triples[k5_idx])
                else:
                    break
                
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Calculate Delta E
                delta_red = 0
                delta_blue = 0
                
                # Update K4 counts
                for k4_idx in triple_to_k4[t_idx]:
                    if old_color == 0:  # Was Red, becoming Blue
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        elif old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        red_count_k4[k4_idx] = new_cnt
                    else:  # Was Blue, becoming Red
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        elif old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        red_count_k4[k4_idx] = new_cnt
                        
                # Update K5 counts
                for k5_idx in triple_to_k5[t_idx]:
                    if old_color == 1:  # Was Blue, becoming Red
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        elif old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        blue_count_k5[k5_idx] = new_cnt
                    else:  # Was Red, becoming Blue
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        elif old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        blue_count_k5[k5_idx] = new_cnt
                        
                delta_E = delta_red + delta_blue
                
                if delta_E < 0:
                    # Accept
                    colors[t_idx] = new_color
                    red_violations += delta_red
                    blue_violations += delta_blue
                    no_improve = 0
                else:
                    # Accept with probability
                    if T > min_T:
                        prob = math.exp(-delta_E / T)
                        if random.random() < prob:
                            colors[t_idx] = new_color
                            red_violations += delta_red
                            blue_violations += delta_blue
                            no_improve += 1
                        else:
                            # Rollback
                            for k4_idx in triple_to_k4[t_idx]:
                                if old_color == 0:
                                    red_count_k4[k4_idx] += 1
                                else:
                                    red_count_k4[k4_idx] -= 1
                            for k5_idx in triple_to_k5[t_idx]:
                                if old_color == 1:
                                    blue_count_k5[k5_idx] += 1
                                else:
                                    blue_count_k5[k5_idx] -= 1
                            no_improve += 1
                    else:
                        # Rollback
                        for k4_idx in triple_to_k4[t_idx]:
                            if old_color == 0:
                                red_count_k4[k4_idx] += 1
                            else:
                                red_count_k4[k4_idx] -= 1
                        for k5_idx in triple_to_k5[t_idx]:
                            if old_color == 1:
                                blue_count_k5[k5_idx] += 1
                            else:
                                blue_count_k5[k5_idx] -= 1
                        no_improve += 1
                
                if red_violations == 0 and blue_violations == 0:
                    return colors
                
                T *= cooling
                
                # Restart if stuck
                if no_improve > patience:
                    no_improve = 0
                    T = max(T, 0.5)
                
                if no_improve < 0:
                    no_improve = 0
            
            # Update best result
            total_viol = red_violations + blue_violations
            if total_viol < best_violations:
                best_violations = total_viol
                best_result = colors[:]
        
        return best_result if best_violations == 0 else None
    
    # Start with a guaranteed valid small n
    n0 = 9
    best_n = n0
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(n0), 3)]
    
    # Try increasing n from 10 up to 34
    for n in range(10, 35):
        if time_left() < 10.0:
            break
        
        result = solve_n(n)
        if result is not None:
            best_n = n
            best_colors = result
        else:
            # If we can't solve n, try n+1 anyway? Or break?
            # Larger n is harder, but sometimes structure helps.
            # Let's try a couple more, but if time is short, stop.
            if time_left() < 15.0:
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
