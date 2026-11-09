import time
import random
import numpy as np

def score_set(S):
    S = set(S)
    diffs = set()
    # Efficient check: for each pair (a, b) with a < b, check if a + 2*(b-a) is in S
    # Or iterate d and check if any a, a+d, a+2d in S.
    # Max d is (max(S)-min(S))//2
    min_s = min(S)
    max_s = max(S)
    for d in range(1, (max_s - min_s) // 2 + 1):
        for a in range(min_s, max_s - 2*d + 1):
            if a in S and (a + d) in S and (a + 2*d) in S:
                diffs.add(d)
                break
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: Interval
    best_set = list(range(1, 33))
    best_score = score_set(best_set)
    
    # Simulated Annealing
    current_set = best_set[:]
    current_score = best_score
    
    T = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Generate neighbor
        i = random.randint(0, 31)
        j = random.randint(0, 31)
        if i == j:
            continue
        
        # Swap
        old_i = current_set[i]
        old_j = current_set[j]
        
        # Try to replace one element with a new one
        # Or swap two elements? Swapping doesn't change the set.
        # We need to change the set.
        # Pick an element x in S, replace with y not in S.
        
        x_idx = random.randint(0, 31)
        x = current_set[x_idx]
        
        # Pick y
        # Bias towards current range or random in 1..512
        if random.random() < 0.5:
            y = random.randint(1, 512)
        else:
            # Local move
            y = x + random.randint(-10, 10)
            y = max(1, min(512, y))
            
        if y in current_set:
            continue
            
        current_set[x_idx] = y
        new_score = score_set(current_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            current_set[x_idx] = x
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_set

