import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # 1. Always include 1
    result = [1]
    
    # 2. Primes > MAX_N // 2 cannot be multiplied by any other prime >= 2
    #    without exceeding MAX_N. So they must be included as themselves.
    isolated_primes = [p for p in primes if p > MAX_N // 2]
    result.extend(isolated_primes)
    
    # 3. The remaining primes are <= MAX_N // 2.
    #    We want to partition them into subsets such that the product of each subset
    #    is <= MAX_N, maximizing the sum of the products.
    #    Greedy Strategy: Repeatedly find the subset with the maximum product <= MAX_N.
    
    pool = [p for p in primes if p <= MAX_N // 2]
    
    # Sort pool descending to help with pruning or just for determinism
    pool.sort(reverse=True)
    
    while pool:
        if time.time() > deadline - 5:
            break
            
        best_subset = None
        best_prod = 0
        
        # Max number of distinct primes in a product <= 256 is 4 (2*3*5*7=210, 2*3*5*7*11>256)
        # So we only need to check subsets of size 1 to 4.
        for size in range(1, 5):
            if len(pool) < size:
                break
            # To optimize, we can break early if the product of the smallest 'size' primes is > 256?
            # No, we want max product.
            for comb in combinations(pool, size):
                prod = 1
                for p in comb:
                    prod *= p
                    if prod > MAX_N:
                        break
                if prod > MAX_N:
                    continue
                if prod > best_prod:
                    best_prod = prod
                    best_subset = comb
                    
        if best_subset is None:
            # Should not happen, as size 1 always works
            best_prod = pool[0]
            best_subset = (pool[0],)
            
        result.append(best_prod)
        
        # Remove used primes from pool
        used_set = set(best_subset)
        pool = [p for p in pool if p not in used_set]
        
    # Ensure return type is list of ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
