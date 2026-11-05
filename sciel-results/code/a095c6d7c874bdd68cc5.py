import time

def construct():
    start_time = time.time()
    LIMIT = 60  # seconds
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N_MAX + 1))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P[k]
    P = [1] * (N_MAX + 1)
    P[1] = 1
    
    for i in range(2, N_MAX + 1):
        p = spf[i]
        x = i // p
        if x % p == 0:
            y = x // p
            if y % p == 0:
                # i has p^3 or higher
                P[i] = P[x] * p
            else:
                # i has exactly p^2
                P[i] = P[x] * p * p
        else:
            # i has exactly p^1
            P[i] = P[x]
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Find n that maximizes P[n] * P[n+1]
    best_n = START
    best_score = -1
    
    # n ranges from 10000 to 1000000
    # P[n+1] must be valid, so max n is 1000000
    for n in range(START, 1000000 + 1):
        score = P[n] * P[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
