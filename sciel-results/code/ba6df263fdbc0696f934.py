import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Target n=34
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build K4 list and K5 list
    k4_list = []
    for c4 in combinations(range(n), 4):
        a, b, c, d = c4
        k4_list.append((
            triple_to_idx[(a, b, c)], 
            triple_to_idx[(a, b, d)], 
            triple_to_idx[(a, c, d)], 
            triple_to_idx[(b, c, d)]
        ))
    
    k5_list = []
    for c5 in combinations(range(n), 5):
        a, b, c, d, e = c5
        k5_list.append(tuple(triple_to_idx[t] for t in [
            (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
            (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
        ]))
        
    num_k4s = len(k4_list)
    num_k5s = len(k5_list)
    
    # Reverse adjacency
    triple_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        for t_idx in k4:
            triple_k4s[t_idx].append(i)
            
    triple_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        for t_idx in k5:
            triple_k5s[t_idx].append(i)

    def count_violations_slow(colors):
        red_k4 = 0
        blue_k5 = 0
        for k4 in k4_list:
            if all(colors[i] == 0 for i in k4):
                red_k4 += 1
        for k5 in k5_list:
            if all(colors[i] == 1 for i in k5):
                blue_k5 += 1
        return red_k4 + blue_k5

    # Try to find a solution for n=34
    found = False
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Initialize colors
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_counts = [0] * num_k4s
        blue_counts = [0] * num_k5s
        
        # Populate initial counts
        for i, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in triple_k4s[i]:
                    red_counts[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_k5s[i]:
                    blue_counts[k5_idx] += 1
                    
        # Calculate initial violations
        total_viol = 0
        for rc in red_counts:
            if rc == 4:
                total_viol += 1
        for bc in blue_counts:
            if bc == 10:
                total_viol += 1
                
        if total_viol == 0:
            return {'n': n, 'colors': colors}
            
        best_viol = total_viol
        best_colors = colors[:]
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        steps = 0
        max_steps = 500000
        
        while time.time() < deadline - 2.0 and total_viol > 0 and steps < max_steps:
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            # Update K4 counts and delta
            for k4_idx in triple_k4s[i]:
                if old_color == 0: # Removing red
                    red_counts[k4_idx] -= 1
                    if red_counts[k4_idx] == 3:
                        delta -= 1 # Violation removed
                else: # Adding red
                    red_counts[k4_idx] += 1
                    if red_counts[k4_idx] == 4:
                        delta += 1 # Violation created
            
            # Update K5 counts and delta
            for k5_idx in triple_k5s[i]:
                if old_color == 1: # Removing blue
                    blue_counts[k5_idx] -= 1
                    if blue_counts[k5_idx] == 9:
                        delta -= 1 # Violation removed
                else: # Adding blue
                    blue_counts[k5_idx] += 1
                    if blue_counts[k5_idx] == 10:
                        delta += 1 # Violation created
            
            # Acceptance
            if delta < 0:
                colors[i] = new_color
                total_viol += delta
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
            elif delta == 0:
                colors[i] = new_color
            else:
                p = math.exp(-delta / T)
                if random.random() < p:
                    colors[i] = new_color
                    total_viol += delta
            
            T *= cooling
            if T < min_T:
                T = min_T
            steps += 1
            
            if best_viol == 0:
                return {'n': n, 'colors': best_colors}
                
        if best_viol == 0:
            return {'n': n, 'colors': best_colors}
            
        # If we got close, restart from best state
        if best_viol < 10:
            colors = best_colors[:]
            # Re-initialize counts for next restart if we decide to continue, 
            # but since we break out of while loop, we go to next restart.
            # To save time, we just let the next restart be random or use best_colors.
            # For simplicity, we just proceed to next restart with random init.
            # Optimization: We could reuse best_colors as start for next SA run.
            
    # If n=34 failed, try n=33
    if time.time() < deadline - 10.0:
        n = 33
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = []
        for c4 in combinations(range(n), 4):
            a, b, c, d = c4
            k4_list.append((
                triple_to_idx[(a, b, c)], 
                triple_to_idx[(a, b, d)], 
                triple_to_idx[(a, c, d)], 
                triple_to_idx[(b, c, d)]
            ))
        
        k5_list = []
        for c5 in combinations(range(n), 5):
            a, b, c, d, e = c5
            k5_list.append(tuple(triple_to_idx[t] for t in [
                (a, b, c), (a, b, d), (a, b, e), (a, c, d), (a, c, e),
                (a, d, e), (b, c, d), (b, c, e), (b, d, e), (c, d, e)
            ]))
            
        num_k4s = len(k4_list)
        num_k5s = len(k5_list)
        
        triple_k4s = [[] for _ in range(num_triples)]
        for i, k4 in enumerate(k4_list):
            for t_idx in k4:
                triple_k4s[t_idx].append(i)
                
        triple_k5s = [[] for _ in range(num_triples)]
        for i, k5 in enumerate(k5_list):
            for t_idx in k5:
                triple_k5s[t_idx].append(i)

        colors = [random.randint(0, 1) for _ in range(num_triples)]
        red_counts = [0] * num_k4s
        blue_counts = [0] * num_k5s
        
        for i, c in enumerate(colors):
            if c == 0:
                for k4_idx in triple_k4s[i]:
                    red_counts[k4_idx] += 1
            else:
                for k5_idx in triple_k5s[i]:
                    blue_counts[k5_idx] += 1
                    
        total_viol = sum(1 for rc in red_counts if rc == 4) + sum(1 for bc in blue_counts if bc == 10)
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        steps = 0
        
        while time.time() < deadline - 2.0 and total_viol > 0 and steps < 300000:
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            for k4_idx in triple_k4s[i]:
                if old_color == 0:
                    red_counts[k4_idx] -= 1
                    if red_counts[k4_idx] == 3:
                        delta -= 1
                else:
                    red_counts[k4_idx] += 1
                    if red_counts[k4_idx] == 4:
                        delta += 1
            
            for k5_idx in triple_k5s[i]:
                if old_color == 1:
                    blue_counts[k5_idx] -= 1
                    if blue_counts[k5_idx] == 9:
                        delta -= 1
                else:
                    blue_counts[k5_idx] += 1
                    if blue_counts[k5_idx] == 10:
                        delta += 1
            
            if delta < 0:
                colors[i] = new_color
                total_viol += delta
            elif delta == 0:
                colors[i] = new_color
            else:
                p = math.exp(-delta / T)
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
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

