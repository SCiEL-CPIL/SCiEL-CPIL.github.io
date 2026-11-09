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
    prime_power = [1] * limit     # p^e where p is smallest prime factor, e is its exponent
    
    primes = []
    
    for i in range(2, limit):
        if smallest_prime[i] == 0:
            # i is prime
            primes.append(i)
            smallest_prime[i] = i
            exp_count[i] = 1
            prime_power[i] = i
        
        for p in primes:
            if p > smallest_prime[i] or p * i >= limit:
                break
            
            smallest_prime[p * i] = p
            if p == smallest_prime[i]:
                exp_count[p * i] = exp_count[i] + 1
                prime_power[p * i] = prime_power[i] * p
                break
            else:
                exp_count[p * i] = 1
                prime_power[p * i] = p
        
        # Compute P[i] from P[i // smallest_prime[i]]
        sp = smallest_prime[i]
        if i % sp == 0:
            if exp_count[i] >= 2:
                # i = sp^e * m where sp doesn't divide m
                # P[i] = sp^e * P[m]
                m = i // (sp ** exp_count[i])
                P[i] = (sp ** exp_count[i]) * P[m]
            else:
                # i = sp * m where sp doesn't divide m (exp_count=1)
                # P[i] = P[m]
                m = i // sp
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
