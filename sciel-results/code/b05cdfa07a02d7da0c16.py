import time
import random
import math

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def generate_smooth_candidates(limit, primes):
    """Generate numbers composed of the given primes up to limit."""
    candidates = set([1])
    for p in primes:
        new_candidates = set()
        for c in candidates:
            val = c
            while val * p <= limit:
                val *= p
                new_candidates.add(val)
        candidates.update(new_candidates)
    return sorted([c for c in candidates if c <= limit])

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_overall_cost = float('inf')
    best_overall_arr = []
    
    # Base primes to try
    base_primes = [2, 3, 5, 7, 11, 13, 17]
    
    # We want to find Y such that X = p*Y has minimal distinct prime factors in sums.
    # Sum X_i + X_j = p*(Y_i + Y_j).
    # Primes in X_i + X_j = {p} union Primes(Y_i + Y_j).
    # So we minimize distinct primes in Y_i + Y_j.
    
    for p in base_primes:
        if time.time() > deadline:
            break
            
        max_y = 10000 // p
        if max_y < 12:
            continue
            
        # Generate initial guess from smooth numbers
        smooths = generate_smooth_candidates(max_y, (2, 3, 5))
        if len(smooths) < 12:
            # Fallback to all integers if not enough smooths
            smooths = list(range(1, max_y + 1))
            
        # Pick 12 distinct smooth numbers as initial guess
        # We want them to be "close" to each other to keep sums small? 
        # Or spread out? 
        # Let's just pick the first 12.
        initial_y = smooths[:12]
        
        current_y = initial_y[:]
        current_cost = count_distinct_primes(current_y)
        
        best_y = current_y[:]
        best_cost = current_cost
        
        T = 10.0
        cooling_rate = 0.9995
        step = 0
        
        while time.time() < deadline:
            step += 1
            if T < 0.01:
                break
            
            # Pick a random index to modify
            idx = random.randint(0, 11)
            old_val = current_y[idx]
            
            # Generate a new value
            # Bias towards smooth numbers? 
            # 50% chance to pick a random smooth number, 50% random integer
            if random.random() < 0.5:
                new_val = random.choice(smooths)
            else:
                new_val = random.randint(1, max_y)
            
            # Check distinctness
            if new_val in current_y and new_val != old_val:
                continue
            
            # Apply move
            current_y[idx] = new_val
            
            # Evaluate
            new_cost = count_distinct_primes(current_y)
            
            # Acceptance criterion
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_y = current_y[:]
            else:
                # Simulated Annealing
                delta = new_cost - current_cost
                if delta > 0 and random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Reject move
                    current_y[idx] = old_val
            
            T *= cooling_rate
            
        # Update global best
        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            # Map back to X
            best_overall_arr = [y * p for y in best_y]
            
    # If we didn't find anything (shouldn't happen), return a default
    if not best_overall_arr:
        return [3 + 6 * k for k in range(12)]
        
    return best_overall_arr
