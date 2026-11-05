"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0 # 10 seconds margin before 90s

    # 1. Sieve for Primes
    is_prime = [True] * (N + 2)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False

    # 2. Compute powerful part for all numbers up to N
    # pp[n] will hold the powerful part of n
    pp = [1] * (N + 2)
    
    # For each prime p, update pp[m] for multiples of p^k
    for p in primes:
        p_power = p * p
        k = 2
        while p_power <= N:
            # Iterate m = j * p_power where j is not divisible by p
            # m <= N => j <= N // p_power
            max_j = N // p_power
            for j in range(1, max_j + 1):
                if j % p != 0:
                    m = j * p_power
                    pp[m] *= p_power
            
            # Check deadline
            if time.time() > deadline:
                return MIN_N # Fallback, though unlikely to happen this fast

            # Next power
            p_power *= p
            k += 1

    # 3. Find max product
    best_n = MIN_N
    best_val = 0

    for n in range(MIN_N, N):
        # Check deadline
        if time.time() > deadline:
            break
        
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
