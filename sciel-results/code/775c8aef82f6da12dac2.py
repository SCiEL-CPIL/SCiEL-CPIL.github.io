"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all numbers up to MAX_N
    # Using recurrence:
    # Let n = p * m where p = spf[n]
    # If m % p != 0: P(n) = P(m)
    # If m % p == 0:
    #   If (m // p) % p != 0: P(n) = P(m) * p * p
    #   Else: P(n) = P(m) * p
    
    pp = [1] * (MAX_N + 2)
    
    for i in range(2, MAX_N + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            # p divides m
            if (m // p) % p == 0:
                # p^2 divides m, so v_p(m) >= 2, v_p(n) >= 3
                pp[i] = pp[m] * p
            else:
                # p divides m but p^2 does not, so v_p(m) = 1, v_p(n) = 2
                pp[i] = pp[m] * p * p
        else:
            # p does not divide m, so v_p(m) = 0, v_p(n) = 1
            pp[i] = pp[m]
        
        # Check deadline periodically
        if i % 100000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Find the best consecutive pair
    best_n = MIN_N
    best_product = 0
    
    for i in range(MIN_N, MAX_N):
        if time.time() > deadline - 2:
            break
        prod = pp[i] * pp[i + 1]
        if prod > best_product:
            best_product = prod
            best_n = i
    
    return best_n
# EVOLVE-BLOCK-END
