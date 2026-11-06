import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Sieve to compute Largest Prime Factor (LPF) for all numbers up to limit
    lpf = [0] * (limit + 1)
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    if time.time() > deadline - 5:
        return N_MIN
        
    best_n = N_MIN
    best_score = float('inf')
    
    # Scan for the n that minimizes max(LPF(n), LPF(n+1))
    for n in range(N_MIN, N_MAX + 1):
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            
    return best_n
