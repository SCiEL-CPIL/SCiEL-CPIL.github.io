import time

def construct():
    start_time = time.time()
    # Use a safety margin, e.g., 50 seconds out of 60/90
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # We need to check n and n+1, so up to N_MAX + 1
    limit = N_MAX + 1
    
    # Sieve of Eratosthenes variant to find Smallest Prime Factor (SPF)
    # spf[i] will store the smallest prime factor of i
    spf = list(range(limit + 1))
    
    # Optimization: only iterate up to sqrt(limit)
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute Largest Prime Factor (LPF) for each number
    # lpf[i] is the largest prime factor of i
    lpf = [0] * (limit + 1)
    lpf[1] = 1
    
    for i in range(2, limit + 1):
        p = spf[i]
        rest = i // p
        # The largest prime factor of i is either p or the largest prime factor of rest
        if lpf[rest] > p:
            lpf[i] = lpf[rest]
        else:
            lpf[i] = p
            
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
            
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        
        # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        # because gcd(n, n+1) = 1, so prime factors are disjoint
        current_max = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
