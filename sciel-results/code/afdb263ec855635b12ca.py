import random
import math
import time

def score_set(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0 # log(1) is 0, division by zero or undefined behavior in context
    
    # Calculate U+U and U-U
    sum_set = set()
    diff_set = set()
    
    # Optimization: if U is small, nested loop is fine
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1.0:
        # If ratio is 1, score is 1. If < 1, score < 1.
        # But we want to maximize.
        pass
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    # Simulated Annealing
    best_set = [0, 1, 2, 3, 7]
    best_score = score_set(best_set)
    
    # Parameters
    max_val = 30
    max_size = 8
    start_temp = 1.0
    cool_rate = 0.95
    iterations = 20000
    start_time = time.time()
    time_limit = 10 # seconds
    
    current_set = list(best_set)
    current_score = best_score
    
    temp = start_temp
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Generate move
        # Toggle a random element
        # Ensure 0 is always in the set
        # Choose a random index from 1 to max_val
        idx = random.randint(1, max_val)
        
        # Check size constraint
        if idx in current_set:
            if len(current_set) <= 1:
                continue
            new_set = [x for x in current_set if x != idx]
        else:
            if len(current_set) >= max_size:
                continue
            new_set = current_set + [idx]
            new_set.sort()
        new_score = score_set(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                
        temp *= cool_rate
        if temp < 1e-6:
            temp = 1e-6
            
    return best_set
