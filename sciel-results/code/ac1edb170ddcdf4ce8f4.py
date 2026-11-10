"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    N = 1000000
    # We need powerful parts for numbers up to N (since n can be up to N, n+1 up to N+1)
    max_val = N + 1
    
    # Step 1: Sieve for smallest prime factor (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute powerful part for each number
    # powerful_part[n] = product of p^e for all prime factors p with exponent e >= 2
    powerful = [1] * (max_val + 1)
    
    for n in range(2, max_val + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        powerful[n] = pp
    
    # Check deadline
    if time.time() > deadline:
        # Shouldn't happen for this size, but safety
        return 10000
    
    # Step 3: Find n in [10000, 1000000] maximizing powerful[n] * powerful[n+1]
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]
    
    for n in range(10000, N + 1):
        val = powerful[n] * powerful[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
