import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    LIMIT = 1000000
    MAX_VAL = LIMIT + 1
    
    # lpf[i] stores the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve to compute largest prime factors
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
                
    best_n = 10000
    best_score = float('inf')
    
    for n in range(10000, LIMIT + 1):
        if time.time() > deadline:
            break
        current_max = max(lpf[n], lpf[n + 1])
        if current_max < best_score:
            best_score = current_max
            best_n = n
            
    return best_n
