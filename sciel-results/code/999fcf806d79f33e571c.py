import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000000
    START = 10000
    
    # Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using SPF
    # Use dynamic programming: P[n] can be derived from P[n//p] if p is the smallest prime factor
    # If n = p^e * m where p doesn't divide m:
    #   If e >= 2: P[n] = p^e * P[m]
    #   If e == 1: P[n] = P[m]
    P = [1] * (N + 1)
    
    for k in range(2, N + 1):
        p = spf[k]
        m = k // p
        if m % p == 0:
            # p divides m, so exponent of p in k is at least 2
            # P[k] = p * P[m] if we know the exponent... 
            # Actually, let's use a different approach:
            # P[k] = P[m] * (p^e if e>=2 else 1) where e is exponent of p in k
            # But we can compute: if m % p == 0, then P[k] = p * P[m] only if the exponent of p in m was already >= 1
            # Let's just use the direct method but optimized
            pass
        # Direct computation with SPF
        x = k
        p_part = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part *= (p ** e)
        P[k] = p_part
    
    # Find n in [10000, N-1] maximizing P[n] * P[n+1]
    best_n = START
    best_val = P[START] * P[START + 1]
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
