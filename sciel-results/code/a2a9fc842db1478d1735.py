"""Search-based program: find (A,B) with 3 distinct ints in [1,128] maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import random
from math import gcd
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    random.seed(42)

    # Precompute primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_all_prime_distinct(vals):
        if len(vals) != 9:
            return False
        return all(is_prime[v] for v in vals)

    best_score = -1
    best_pair = ([1, 7, 13], [4, 10, 16]) # Seed fallback (score 5)

    # Strategy: Search for 9 distinct prime sums.
    # This guarantees pairwise coprimality.
    
    for iteration in range(2000000):
        if time.time() > deadline:
            break
        
        # Generate random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        s = distinct_sums(A, B)
        if len(s) != 9:
            continue
        
        if is_all_prime_distinct(s):
            best_score = 9
            best_pair = (A, B)
            break
            
    # Fallback to seed if no prime solution found (unlikely)
    if best_score < 0:
        final_A, final_B = [1, 7, 13], [4, 10, 16]
    else:
        final_A, final_B = best_pair
        
    return [final_A, final_B]
# EVOLVE-BLOCK-END
