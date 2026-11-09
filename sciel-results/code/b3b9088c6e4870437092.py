import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Linear sieve to compute powerful part P(x) for all x in [0, limit)
    # P(x) = product of p^e where e >= 2 in factorization of x
    
    P = [1] * limit
    smallest_prime = [0] * limit  # smallest prime factor
    exp_count = [0] * limit       # exponent of smallest prime factor
    ppow = [1] * limit            # highest power of smallest prime factor dividing x
    
    primes = []
    
    for i in range(2, limit):
        if smallest_prime[i] == 0:
            # i is prime
            primes.append(i)
            smallest_prime[i] = i
            exp_count[i] = 1
            ppow[i] = i
        
        for p in primes:
            if p > smallest_prime[i] or p * i >= limit:
                break
            
            j = p * i
            smallest_prime[j] = p
            if p == smallest_prime[i]:
                exp_count[j] = exp_count[i] + 1
                ppow[j] = ppow[i] * p
                break
            else:
                exp_count[j] = 1
                ppow[j] = p
        
        # Compute P[i]
        # m is i with all factors of smallest_prime[i] removed
        m = i // ppow[i]
        
        if exp_count[i] >= 2:
            P[i] = ppow[i] * P[m]
        else:
            P[i] = P[m]
            
    # Find max P[n] * P[n+1]
    best_n = N_MIN
    best_val = P[N_MIN] * P[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > DEADLINE:
            break
            
    return best_n
