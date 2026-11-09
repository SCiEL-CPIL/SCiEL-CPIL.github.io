import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need pp[n] and pp[n+1] for n up to N, so indices up to N+1
    
    # Linear sieve to compute powerful parts in O(N)
    lp = [0] * (MAX_VAL + 1)      # smallest prime factor
    cnt = [0] * (MAX_VAL + 1)     # exponent of lp[i] in i
    rest = [1] * (MAX_VAL + 1)    # i / lp[i]^cnt[i]
    pp = [1] * (MAX_VAL + 1)      # powerful part of i
    
    primes = []
    
    for i in range(2, MAX_VAL + 1):
        if lp[i] == 0:
            lp[i] = i
            cnt[i] = 1
            rest[i] = 1
            pp[i] = 1  # prime, no powerful part
            primes.append(i)
        
        for p in primes:
            if p > lp[i] or i * p > MAX_VAL:
                break
            ip = i * p
            lp[ip] = p
            
            if p == lp[i]:
                cnt[ip] = cnt[i] + 1
                rest[ip] = rest[i]
                if cnt[i] >= 2:
                    # p^cnt[i] already in pp[i], now p^(cnt[i]+1), multiply by p
                    pp[ip] = pp[i] * p
                else:
                    # cnt[i] == 1, p^1 not in pp[i], now p^2, multiply by p^2
                    pp[ip] = pp[i] * p * p
                break
            else:
                # p < lp[i], so p is a new smaller prime factor
                cnt[ip] = 1
                rest[ip] = i
                pp[ip] = pp[i]  # p^1 doesn't contribute to powerful part
    
    # Find n in [10000, 1000000] that maximizes pp[n] * pp[n+1]
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, N + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
