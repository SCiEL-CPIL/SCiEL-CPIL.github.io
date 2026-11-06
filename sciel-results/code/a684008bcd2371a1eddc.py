import time
import random
import math

def compute_metrics(A):
    A = list(A)
    sum_set = set()
    prod_set = set()
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Initial set: 1..20
    A = list(range(1, 21))
    best_score, _, _ = compute_metrics(A)
    best_A = A[:]
    
    # Candidate pool
    # Include small numbers, smooth numbers, and some random large numbers
    pool = set(range(1, 101))
    # Add powers of 2, 3, 5
    for base in [2, 3, 5, 7]:
        val = 1
        while val <= 10000:
            pool.add(val)
            val *= base
    # Add random numbers
    for _ in range(500):
        pool.add(random.randint(1, 10000))
    pool = list(pool)
    
    current_A = A[:]
    current_score, _, _ = compute_metrics(current_A)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to replace
        i = random.randint(0, 19)
        # Pick a random candidate
        c = random.choice(pool)
        
        if c in current_A:
            continue
            
        # Create new set
        new_A = current_A[:]
        new_A[i] = c
        
        # Compute new score
        new_score, _, _ = compute_metrics(new_A)
        
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Simulated annealing acceptance
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
        # Occasionally restart current_A from best_A to escape local minima
        if random.random() < 0.01:
            current_A = best_A[:]
            current_score, _, _ = compute_metrics(current_A)
            
    return best_A
