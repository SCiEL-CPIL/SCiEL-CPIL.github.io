import time
import random
import math

def calculate_score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1/1)/log(1) is undefined, but limit is 1? 
                   # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. Log 1 = 0. Score 1.
    
    sumset = set()
    diffset = set()
    
    # Optimization: use lists for speed if small
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        # If ratio <= 1, log is negative or zero, score <= 1
        # But we want to maximize, so we avoid these
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial good set
    U = [0, 1, 3, 5, 9, 13]
    best_score = calculate_score(U)
    best_U = U.copy()
    
    current_U = U.copy()
    current_score = best_score
    
    # Temperature for simulated annealing
    T = 0.01
    T_min = 1e-6
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor = current_U.copy()
        move = random.choice(['add', 'remove', 'shift'])
        
        if move == 'add':
            # Add a number near the current range
            max_val = max(neighbor) if neighbor else 0
            min_val = min(neighbor) if neighbor else 0
            new_val = random.randint(0, max_val + 5)
            if new_val not in neighbor:
                neighbor.append(new_val)
        
        elif move == 'remove':
            if len(neighbor) > 1: # Keep 0
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                
        elif move == 'shift':
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                delta = random.choice([-2, -1, 1, 2])
                new_val = neighbor[idx] + delta
                if new_val >= 0 and new_val not in neighbor:
                    neighbor[idx] = new_val
        
        neighbor.sort()
        new_score = calculate_score(neighbor)
        
        # Acceptance probability
        if new_score > current_score:
            current_U = neighbor
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = neighbor.copy()
        else:
            # Simulated annealing acceptance
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_U = neighbor
                current_score = new_score
                
        # Cool down
        T *= 0.999
        
        # If T is too low, restart with best_U
        if T < T_min:
            current_U = best_U.copy()
            current_score = best_score
            T = 0.01
            
    return best_U
