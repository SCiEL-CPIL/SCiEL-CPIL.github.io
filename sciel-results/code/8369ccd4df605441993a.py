import math

def construct():
    N = 1000001
    lpf = [1] * (N + 1)
    
    # Sieve for largest prime factor
    # We iterate i from 2 to sqrt(N)
    # If lpf[i] == 1, i is prime.
    # Then for all multiples j of i, set lpf[j] = i.
    # Since i increases, the last prime factor assigned is the largest.
    for i in range(2, int(math.isqrt(N)) + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Primes > sqrt(N) remain 1. Set them to themselves.
    for i in range(int(math.isqrt(N)) + 1, N + 1):
        if lpf[i] == 1:
            lpf[i] = i
            
    min_val = float('inf')
    best_n = 10000
    
    for n in range(10000, 1000001):
        val = max(lpf[n], lpf[n+1])
        if val < min_val:
            min_val = val
            best_n = n
            
    return best_n
