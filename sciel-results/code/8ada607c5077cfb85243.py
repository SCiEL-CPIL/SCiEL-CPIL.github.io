import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 20
    max_val = 10000
    
    # Initial set: Arithmetic Progression
    best_set = list(range(1, n + 1))
    best_score = get_score(best_set)
    
    # Simulated Annealing
    current_set = best_set.copy()
    current_score = best_score
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Precompute valid range for random selection
    # We pick a random index and a random value
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Choose a random index to replace
        i = random.randint(0, n - 1)
        old_val = current_set[i]
        
        # Choose a new value
        # To avoid collisions, we can pick a random value and check, or retry
        # Given n=20, collision probability is low (20/10000)
        new_val = random.randint(1, max_val)
        
        # Ensure distinctness
        if new_val in current_set:
            continue
            
        # Apply move
        current_set[i] = new_val
        
        # Evaluate
        new_score = get_score(current_set)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set.copy()
        else:
            # Revert
            current_set[i] = old_val
            
        T *= cooling_rate

    return best_set

def get_score(A):
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))
