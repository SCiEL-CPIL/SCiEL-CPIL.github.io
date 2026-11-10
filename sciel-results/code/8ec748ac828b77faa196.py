import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin
    
    def score(A):
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

    # Generate pool of smooth numbers (2,3,5,7-smooth) <= 10000
    primes = [2, 3, 5, 7]
    pool = []
    for i in range(1, 10001):
        x = i
        for p in primes:
            while x % p == 0:
                x //= p
        if x == 1:
            pool.append(i)
            
    # Ensure pool has enough elements
    if len(pool) < 100:
        pool = list(range(1, 10001))

    best_set = None
    best_score = float('inf')
    
    def sa_search(initial_set):
        nonlocal best_set, best_score
        # Ensure initial set is valid (distinct, <= 10000)
        # The candidates below are valid.
        current = initial_set[:]
        current_score = score(current)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
            
        T = 5.0
        cooling = 0.995
        
        while time.time() - start_time < timeout:
            idx = random.randint(0, 19)
            old_val = current[idx]
            
            # Pick new value from pool
            new_val = random.choice(pool)
            
            # Check distinctness
            if new_val == old_val:
                continue
            if new_val in current:
                # Simple retry or skip
                # To avoid infinite loop on small pools, just skip
                continue
                
            new_set = current[:]
            new_set[idx] = new_val
            new_score = score(new_set)
            
            delta = new_score - current_score
            
            if delta < 0:
                current = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                # Avoid division by zero or overflow in exp
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    current = new_set
                    current_score = new_score
            
            T *= cooling
            if T < 0.5:
                T = 0.5

    # Initial Candidates
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # 2. Odd Arithmetic Progression
    candidates.append(list(range(1, 41, 2)))
    
    # 3. Geometric Progression (Powers of 2)
    # 2^19 is 524288, which is > 10000. So we can only go up to 2^13=8192.
    # We need 20 distinct. So GP is not a valid full candidate if we stick to powers of 2.
    # We can use a mix or just skip this specific one if it violates constraints.
    # Let's use a GP that fits: 1, 2, 4, ..., 8192 (14 terms). Not enough.
    # So GP is not a good initial candidate for n=20, limit=10000.
    
    # 4. First 20 Smooth Numbers
    candidates.append(pool[:20])
    
    # 5. Random Smooth Numbers
    for _ in range(3):
        random_set = random.sample(pool, 20)
        candidates.append(random_set)
        
    # 6. Random Uniform
    for _ in range(3):
        random_set = random.sample(range(1, 10001), 20)
        candidates.append(random_set)

    # Run SA on each candidate, but stop if time is up
    for cand in candidates:
        if time.time() - start_time > timeout:
            break
        sa_search(cand)
        
    return best_set
