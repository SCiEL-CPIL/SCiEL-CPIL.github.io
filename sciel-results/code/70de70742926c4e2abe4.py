import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # 1000001
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
                
    best_n = N_MIN
    best_val = lpf[N_MIN] * lpf[N_MIN+1] # Initialize with a large value or just compute first
    
    # Actually, let's initialize best_val to infinity
    best_val = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        # Check time
        if time.time() > deadline:
            break
            
        v1 = lpf[n]
        v2 = lpf[n+1]
        current_max = v1 if v1 > v2 else v2
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
