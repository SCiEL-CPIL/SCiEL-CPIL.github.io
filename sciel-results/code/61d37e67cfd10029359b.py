import time

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    limit = 1000000
    max_val = limit + 1
    
    # Initialize largest prime factor array
    lpf = [0] * (max_val + 1)
    
    # Sieve to find largest prime factor
    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
    
    # Check time after sieve
    if time.time() - start_time > deadline - 10:
        return 10000
    
    best_n = 10000
    min_score = float('inf')
    
    # Scan for the pair with minimum max LPF
    for n in range(10000, limit + 1):
        score = max(lpf[n], lpf[n+1])
        if score < min_score:
            min_score = score
            best_n = n
    
    return best_n
