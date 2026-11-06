"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Precompute smallest prime factor (SPF) for efficient factorization
    MAX_N = 1000000
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    def powerful_part(n):
        if n <= 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                result *= (p ** count)
        return result
    
    # Precompute powerful parts for all numbers
    pp = [0] * (MAX_N + 1)
    pp[0] = 1
    pp[1] = 1
    for i in range(2, MAX_N + 1):
        pp[i] = powerful_part(i)
    
    # Find the best n in [10000, 999999] (since n+1 must be <= 1000000)
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, 1000000):
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
