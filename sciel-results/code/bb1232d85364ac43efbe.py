"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s out of 90s limit for safety margin
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Compute smallest prime factor for all numbers up to N_MAX
    spf = list(range(N_MAX + 1))
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part(n) = product of p^e where e >= 2 in factorization of n
    pow_part = [1] * (N_MAX + 1)
    
    for n in range(2, N_MAX + 1):
        temp = n
        pp = 1
        while temp > 1:
            p = spf[temp]
            # Count exponent of p
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                pp *= (p ** exp)
        pow_part[n] = pp
    
    # Find the n in [10000, 1000000-1] that maximizes pow_part[n] * pow_part[n+1]
    # Note: n+1 must be <= 1000000, so n <= 999999
    best_n = N_MIN
    best_product = 0
    
    for n in range(N_MIN, N_MAX):  # n from 10000 to 999999
        prod = pow_part[n] * pow_part[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
