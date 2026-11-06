import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    N = 1000000
    # Precompute powerful_part for all numbers 1..N+1
    pp = [1] * (N + 2)  # index 0..N+1
    
    # SPF sieve
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful part for each number using SPF
    for m in range(2, N + 2):
        x = m
        result = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                result *= p ** e
        pp[m] = result
    
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10001, N):
        if time.time() - start_time > DEADLINE:
            break
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
    
    return best_n
