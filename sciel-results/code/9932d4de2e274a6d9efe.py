import time
import random
import math

# Precompute smallest prime factor for numbers up to 20000
MAX_VAL = 20000
spf = list(range(MAX_VAL + 1))
for i in range(2, int(MAX_VAL**0.5) + 1):
    if spf[i] == i:  # i is prime
        for j in range(i*i, MAX_VAL + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    """Return a set of prime factors of n."""
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    """Count distinct prime factors of all pairwise sums."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            prime_set.update(get_prime_factors(s))
    return len(prime_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe
    
    # Initial candidates
    # Strategy: Start with multiples of small highly composite numbers
    # Multiples of 6
    init1 = list(range(6, 6*13, 6))
    # Multiples of 12
    init2 = list(range(12, 12*13, 12))
    # Multiples of 30
    init3 = list(range(30, 30*13, 30))
    
    best_arr = init1
    best_score = count_distinct_primes(best_arr)
    
    # Also try some random starts biased towards multiples of 6
    for _ in range(10):
        rand_arr = random.sample(range(1, 10001), 12)
        # Bias: replace some with multiples of 6
        for i in range(12):
            if random.random() < 0.5:
                rand_arr[i] = random.choice(range(6, 10001, 6))
        score = count_distinct_primes(rand_arr)
        if score < best_score:
            best_score = score
            best_arr = rand_arr[:]

    # Local Search
    # Candidate values for replacement:
    # Focus on numbers that are multiples of 2, 3, 5, 6, 10, 12, 30, etc.
    # This increases the chance that sums share factors.
    candidate_values = []
    for m in [2, 3, 5, 6, 10, 12, 15, 20, 30, 60]:
        for x in range(m, 10001, m):
            if x not in candidate_values:
                candidate_values.append(x)
    # Add some random numbers
    candidate_values.extend(random.sample(range(1, 10001), 1000))
    
    current_arr = best_arr[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # Try to improve each position
        for i in range(12):
            if time.time() > deadline:
                break
            
            old_val = current_arr[i]
            # Try a few random candidates from the pool
            # We can also try nearby values
            candidates = random.sample(candidate_values, 20)
            # Ensure old_val is not in candidates if we want to force change, 
            # but allowing same value is fine for stability, though pointless.
            
            best_candidate_val = old_val
            best_candidate_score = current_score
            
            for v in candidates:
                if v == old_val:
                    continue
                # Check distinctness
                if v in current_arr[:i] or v in current_arr[i+1:]:
                    continue
                
                new_arr = current_arr[:]
                new_arr[i] = v
                new_score = count_distinct_primes(new_arr)
                
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate_val = v
            
            if best_candidate_val != old_val:
                current_arr[i] = best_candidate_val
                current_score = best_candidate_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_arr = current_arr[:]
        
        if not improved:
            # If stuck, try a more aggressive random perturbation
            # or just break if we've been stuck for a while?
            # Let's do a random restart of the local search part of the array
            if random.random() < 0.1:
                # Perturb 2-3 elements randomly
                for _ in range(random.randint(2, 3)):
                    idx = random.randint(0, 11)
                    new_val = random.choice(candidate_values)
                    # Ensure distinct
                    attempts = 0
                    while new_val in current_arr and attempts < 10:
                        new_val = random.choice(candidate_values)
                        attempts += 1
                    if new_val not in current_arr:
                        current_arr[idx] = new_val
                current_score = count_distinct_primes(current_arr)
                if current_score < best_score:
                    best_score = current_score
                    best_arr = current_arr[:]

    return best_arr
