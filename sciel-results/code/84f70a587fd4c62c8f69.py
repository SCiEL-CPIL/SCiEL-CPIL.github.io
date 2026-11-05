import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N = 1000000
    MAX_VAL = N + 24
    
    # Sieve for smallest prime factor to speed up factorization
    # We only care about primes in 'primes', but SPF helps us strip them quickly
    # Actually, we can just check divisibility by the 9 primes directly.
    # 9 checks per number * 24 numbers * 1e6 iterations = 216e6 ops.
    # Python might be too slow for 216e6 ops in 50s.
    # Let's optimize.
    
    # Alternative: Precompute the "reduced factor" for all numbers up to MAX_VAL?
    # Memory: 1e6 integers. Time: 1e6 * 9 checks = 9e6 ops. Very fast.
    
    reduced_factor = [1] * (MAX_VAL + 1)
    for p in primes:
        for multiple in range(p, MAX_VAL + 1, p):
            reduced_factor[multiple] *= p
            
    best_n = 1000
    best_score = 100 # Initialize with a high number
    
    # We can scan in strides or just linearly.
    # Linear scan is 1e6 iterations.
    # Inside loop: 24 lookups and set creation.
    # 1e6 * 24 = 24e6 lookups. Should be well within 50s in Python.
    
    # Let's iterate
    # To be safe, we can break early if we find score 1 or 2, but let's see how low it goes.
    # The minimum possible is 1 (if all n+i have the same reduced factor).
    # Is it possible? n+i must all be divisible by the same set of primes.
    # If reduced factor is 2, all n+i must be even. Impossible for consecutive integers.
    # If reduced factor is 6, all n+i must be divisible by 2 and 3.
    # n+1, n+2, n+3... one of any 3 consecutive is divisible by 3.
    # But they don't all have to be divisible by 3, they just have to have the *same* reduced factor.
    # Wait, reduced factor is the product of primes < 24 dividing the number.
    # If n+1 is divisible by 2 but not 3, red=2.
    # If n+2 is divisible by 2 and 3, red=6.
    # These are distinct.
    # So we want the set of reds to be as small as possible.
    
    for n in range(1000, N + 1):
        if time.time() > deadline:
            break
            
        # Calculate reduced factors for n+1 to n+24
        # Using a set to count distinct values
        # Optimization: use a local list and set
        vals = set()
        # n+1 to n+24
        # Accessing reduced_factor list is fast
        for i in range(1, 25):
            vals.add(reduced_factor[n + i])
            
        score = len(vals)
        if score < best_score:
            best_score = score
            best_n = n
            
        # If we find a very low score, we might want to continue to see if we can go lower,
        # but 1 is the theoretical minimum. 2 is very good.
        # Let's just keep the minimum.
        
    return best_n
# EVOLVE-BLOCK-END
