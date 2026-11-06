import time
import math

def construct():
    deadline = time.time() + 75
    limit = 1000000
    lower = 10000
    
    # Linear sieve to compute powerful parts
    # spf: smallest prime factor
    # cnt: exponent of spf in the number
    # rest: number / spf^cnt
    # pp: powerful part
    
    N = limit
    spf = [0] * (N + 1)
    cnt = [0] * (N + 1)
    rest = [0] * (N + 1)
    pp = [1] * (N + 1)
    primes = []
    
    # Initialize 1
    spf[1] = 1
    cnt[1] = 0
    rest[1] = 1
    pp[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            cnt[i] = 1
            rest[i] = 1
            pp[i] = 1  # Prime itself has exponent 1, so powerful part is 1
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            
            spf[i * p] = p
            
            if p == spf[i]:
                cnt[i * p] = cnt[i] + 1
                rest[i * p] = rest[i]
                # cnt[i*p] is at least 2
                pp[i * p] = pp[rest[i * p]] * (p ** cnt[i * p])
            else:
                cnt[i * p] = 1
                rest[i * p] = i
                pp[i * p] = pp[i]
                
    # Scan for best consecutive pair
    best_n = lower
    best_val = 0
    for n in range(lower, limit):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
            
    return best_n
