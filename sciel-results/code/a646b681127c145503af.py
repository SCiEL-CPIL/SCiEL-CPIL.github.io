"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Sieve of Eratosthenes for primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_set = set(primes)
    
    # Precompute valid B candidates for each A
    # We want A+B to contain only primes (to ensure pairwise coprime)
    # And we want 9 distinct sums.
    
    # Iterate over all combinations of A
    # C(128, 3) = 341,376
    # For each A, find intersection of (Primes - a) for a in A
    
    # To speed up, we can precompute for each a in [1, 128] the set of b in [1, 128]
    # such that a+b is prime.
    valid_b_for_a = [set() for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            if a + b in prime_set:
                valid_b_for_a[a].add(b)
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Fallback seed
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    
    for A in combinations(range(1, 129), 3):
        if time.time() > deadline:
            break
        
        # Intersection of valid b's for all a in A
        # valid_b_intersection = valid_b_for_a[A[0]] & valid_b_for_a[A[1]] & valid_b_for_a[A[2]]
        valid_bs = valid_b_for_a[A[0]] & valid_b_for_a[A[1]] & valid_b_for_a[A[2]]
        
        if len(valid_bs) < 3:
            continue
        
        # Try to find a B of size 3 from valid_bs such that sums are distinct
        # We can just pick the first 3 and check? No, we need to check distinctness.
        # But if we pick any 3 distinct b's, are the sums guaranteed to be distinct?
        # Sums are a_i + b_j.
        # Collision if a_i + b_j = a_k + b_l => a_i - a_k = b_l - b_j.
        # Since A has distinct elements, a_i - a_k != 0.
        # If we choose b_j such that no differences match, we are good.
        # However, checking all C(|valid_bs|, 3) might be expensive if |valid_bs| is large.
        # But usually |valid_bs| is small.
        
        # Optimization: If we find ANY valid B with 9 distinct sums, we return it immediately
        # because 9 is the maximum possible.
        
        for B in combinations(valid_bs, 3):
            sums = set()
            distinct = True
            for a in A:
                for b in B:
                    s = a + b
                    if s in sums:
                        distinct = False
                        break
                    sums.add(s)
                if not distinct:
                    break
            
            if distinct:
                return [list(A), list(B)]
    
    # If no prime solution found, fall back to seed or random search
    # Given the density of primes, this is highly unlikely.
    
    # Fallback to seed
    return [seed_A, seed_B]
# EVOLVE-BLOCK-END
