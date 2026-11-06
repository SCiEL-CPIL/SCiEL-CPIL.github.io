import time
import random
import math

def compute_metrics(A):
    """
    Computes the size of the sumset and product set of A.
    Returns (max_size, sum_size, prod_size).
    """
    A = list(A)
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sum_set.add(ai + A[j])
            prod_set.add(ai * A[j])
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Generate a diverse candidate pool
    # We want numbers that are "smooth" or small to encourage collisions
    pool = set()
    # Small integers
    for i in range(1, 500):
        pool.add(i)
    # Powers of small primes
    for base in [2, 3, 5, 7, 11]:
        val = 1
        while val <= 10000:
            pool.add(val)
            val *= base
    # Random smooth numbers (product of small primes)
    small_primes = [2, 3, 5, 7, 11, 13]
    for _ in range(2000):
        val = 1
        for _ in range(random.randint(1, 5)):
            val *= random.choice(small_primes)
        if val <= 10000:
            pool.add(val)
    pool = list(pool)
    
    best_score = float('inf')
    best_A = []
    
    # 2. Multi-Start Local Search
    # We will run local search from several different starting points
    
    # Initial configurations
    initial_sets = []
    # Standard 1..20
    initial_sets.append(list(range(1, 21)))
    # First 20 smooth numbers (sorted)
    smooths = sorted([x for x in pool if x <= 1000])[:20]
    if len(smooths) == 20:
        initial_sets.append(smooths)
    # Random small set
    initial_sets.append(sorted(random.sample(range(1, 100), 20)))
    # Mix of small and powers of 2
    mix = list(range(1, 11)) + [2**i for i in range(4, 14)]
    # Filter to distinct and <= 10000
    mix = [x for x in mix if x <= 10000]
    # If we need more, add random smalls
    while len(mix) < 20:
        c = random.randint(1, 50)
        if c not in mix:
            mix.append(c)
    initial_sets.append(mix[:20])
    
    for init_idx, init_A in enumerate(initial_sets):
        # Ensure validity: 20 distinct positive integers <= 10000
        if len(set(init_A)) != 20 or any(x > 10000 for x in init_A):
            # Fallback to 1..20 if invalid
            init_A = list(range(1, 21))
        
        current_A = init_A[:]
        current_score, _, _ = compute_metrics(current_A)
        local_best_score = current_score
        local_best_A = current_A[:]
        
        # Local Search Loop for this start
        # Allocate time based on remaining budget
        # Simpler: just run until global deadline, but track best
        while time.time() < deadline:
            # Pick a random index to change
            i = random.randint(0, 19)
            # Pick a candidate
            # Bias towards numbers close to current A[i] or from the pool
            c = random.choice(pool)
            
            if c in current_A:
                continue
                
            new_A = current_A[:]
            new_A[i] = c
            new_score, _, _ = compute_metrics(new_A)
            
            # Acceptance criteria: Accept if better, or with prob exp(-delta/T)
            # Use a simple temperature that cools down
            # We don't have a global T, so let's use a simple hill climbing with occasional random jumps
            if new_score < current_score:
                current_A = new_A
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_A = current_A[:]
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
            else:
                # Occasionally accept worse moves to escape local minima
                # Probability decreases as score difference increases
                delta = new_score - current_score
                if delta < 5 and random.random() < 0.1:
                    current_A = new_A
                    current_score = new_score
                elif random.random() < 0.01:
                    # Random restart from local best
                    current_A = local_best_A[:]
                    current_score = local_best_score
            
            # Check time inside loop
            if time.time() > deadline:
                break
        
        # Update global best
        if local_best_score < best_score:
            best_score = local_best_score
            best_A = local_best_A[:]
            
        if time.time() > deadline:
            break
            
    return best_A
