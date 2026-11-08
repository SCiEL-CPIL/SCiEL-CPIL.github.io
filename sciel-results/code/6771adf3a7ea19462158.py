import time
import math
import random

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    sum_set = set()
    diff_set = set()
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial set: Mersenne numbers (2^k - 1) starting from 0 (2^0-1=0)
    # 0, 1, 3, 7, 15
    current_U = [0, 1, 3, 7, 15]
    current_score = compute_score(current_U)
    best_U = list(current_U)
    best_score = current_score
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        neighbor = list(current_U)
        move_type = random.randint(0, 2)
        
        if move_type == 0 and len(neighbor) < 60: # Add
            max_val = neighbor[-1] if neighbor else 1
            new_val = random.randint(1, max(2, 2 * max_val))
            if new_val not in neighbor:
                neighbor.append(new_val)
                neighbor.sort()
        elif move_type == 1 and len(neighbor) > 2: # Remove
            idx = random.randint(1, len(neighbor) - 1)
            neighbor.pop(idx)
        elif move_type == 2: # Change
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                max_val = neighbor[-1]
                new_val = random.randint(1, max(2, 2 * max_val))
                if new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor.sort()
        
        if 0 not in neighbor:
            neighbor = [0] + [x for x in neighbor if x > 0]
            neighbor.sort()
        
        if len(neighbor) < 2:
            continue
            
        neighbor_score = compute_score(neighbor)
        
        delta = neighbor_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_U = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_U
