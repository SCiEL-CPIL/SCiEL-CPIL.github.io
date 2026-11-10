"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    
    # Sieve for primes
    is_prime = [True] * (MAX_N + 2)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, MAX_N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
                
    # Compute powerful part for all n using a sieve-like method
    # pp[n] will store the powerful part of n
    pp = [1] * (MAX_N + 2)
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
            
        # For each power p^k, we identify numbers where v_p(n) == k
        # and multiply their pp by p^k
        k = 2
        pk = p2
        while pk <= MAX_N:
            next_pk = pk * p
            # Iterate through multiples of pk
            # If a multiple m is NOT divisible by next_pk, then v_p(m) is exactly k
            for m in range(pk, MAX_N + 1, pk):
                if m % next_pk != 0:
                    pp[m] *= pk
            k += 1
            pk = next_pk
            
    best_n = MIN_N
    best_product = 0
    
    # Find the maximum product of powerful parts for consecutive integers
    for n in range(MIN_N, MAX_N):
        prod = pp[n] * pp[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
