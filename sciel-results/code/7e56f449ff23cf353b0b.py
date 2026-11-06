import time
import random

# EVOLVE-BLOCK-START

MAX_SUM = 20000
prime_factors_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    for i in range(2, MAX_SUM + 1):
        if not prime_factors_cache[i]:
            for j in range(i, MAX_SUM + 1, i):
                prime_factors_cache[j].add(i)

_init_prime_factors()

def _objective(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(prime_factors_cache[s])
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 12
    max_val = 10000
    
    # Use modulus 210 to exclude 3, 5, 7 from factors
    # a_i = 1 mod 210 => a_i + a_j = 2 mod 210
    # gcd(2, 210) = 2. So 2 is factor. 3, 5, 7 are not.
    candidates = [x for x in range(1, max_val + 1) if x % 210 == 1]
    
    if len(candidates) < n:
        candidates = list(range(1, max_val + 1))
        
    rng = random.Random(42)
    
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        # Initialize with random sample from candidates
        if len(candidates) >= n:
            current = rng.sample(candidates, n)
        else:
            current = list(candidates)
            # If candidates < n, we can't satisfy distinctness from this set only.
            # But 210 modulus gives ~47 candidates, so this is fine.
            
        current_primes = _objective(current)
        current_count = len(current_primes)
        
        if current_count < best_count:
            best_count = current_count
            best_set = list(current)
            best_primes = current_primes
            
        # Local Search
        no_improve = 0
        while time.time() < deadline and no_improve < 500:
            i = rng.randrange(n)
            old_val = current[i]
            
            # Pick new val from candidates
            # To ensure we explore, pick random
            new_val = rng.choice(candidates)
            if new_val in current:
                continue
                
            current[i] = new_val
            new_primes = _objective(current)
            new_count = len(new_primes)
            
            if new_count < current_count:
                current_primes = new_primes
                current_count = new_count
                no_improve = 0
                if current_count < best_count:
                    best_count = current_count
                    best_set = list(current)
                    best_primes = current_primes
            else:
                # Revert
                current[i] = old_val
                no_improve += 1
                
    return best_set

# EVOLVE-BLOCK-END
