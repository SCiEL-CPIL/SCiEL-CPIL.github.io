import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    LIMIT = N + 1  # We need powerful part of N+1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 90s limit (conservative)
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    # Initialize spf with itself
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Precompute powerful part for all numbers up to LIMIT
    # pp[i] = powerful part of i
    pp = [1] * (LIMIT + 1)
    pp[1] = 1
    
    for i in range(2, LIMIT + 1):
        if spf[i] == i:
            # Prime, powerful part is 1
            pp[i] = 1
        else:
            p = spf[i]
            j = i
            cnt = 0
            while j % p == 0:
                j //= p
                cnt += 1
            
            # j is i with all factors of p removed
            # pp[j] is already computed because j < i
            if cnt >= 2:
                pp[i] = (p ** cnt) * pp[j]
            else:
                pp[i] = pp[j]
                
    # Check time
    if time.time() > deadline:
        # Fallback to a simple guess if precomputation took too long (unlikely)
        # Return a known good candidate or the start of the range
        # A powerful number like 10000 (2^4 * 5^4) is a good start
        # But 10000 is not necessarily the max. 
        # Since precomputation is O(N) and N=1e6, this should never trigger.
        return MIN_N

    # 3. Find max product
    best_n = MIN_N
    best_val = 0
    
    # We need P(n) and P(n+1)
    # n ranges from MIN_N to N
    # So we access pp[n] and pp[n+1]
    # pp is computed up to LIMIT = N+1
    
    for n in range(MIN_N, N + 1):
        # Check deadline
        if time.time() > deadline:
            break
            
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
