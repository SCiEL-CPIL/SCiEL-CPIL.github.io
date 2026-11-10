import numpy as np
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0 # Safety margin
    
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.array(np.nonzero(is_prime))[0]
    N = len(primes)
    
    if N == 0:
        return []
        
    max_gap = primes[-1] - primes[0]
    
    # DP Arrays
    best = np.zeros((N, max_gap + 1), dtype=np.int32)
    M = np.zeros((N, max_gap + 1), dtype=np.int32)
    argM = np.zeros((N, max_gap + 1), dtype=np.int32)
    parent = np.full((N, max_gap + 1), -1, dtype=np.int32)
    
    # Base case: sequence of length 1
    best[:, 0] = 1
    M[:, 0] = 1
    argM[:, 0] = 0
    
    global_max_len = 1
    global_max_idx = 0
    global_max_gap = 0
    
    for i in range(N):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        for j in range(i):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            d = primes[i] - primes[j]
            new_len = M[j][d] + 1
            
            if new_len > best[i][d]:
                best[i][d] = new_len
                parent[i][d] = j
                if new_len > global_max_len:
                    global_max_len = new_len
                    global_max_idx = i
                    global_max_gap = d
        
        # Compute M[i] and argM[i]
        M[i] = np.maximum.accumulate(best[i])
        
        current_max = 0
        current_arg = 0
        for g in range(max_gap + 1):
            if best[i][g] > current_max:
                current_max = best[i][g]
                current_arg = g
            argM[i][g] = current_arg
            
    # Backtrack
    seq = []
    i = global_max_idx
    g = global_max_gap
    
    while i != -1:
        seq.append(primes[i])
        if g == 0:
            break
        j = parent[i][g]
        if j == -1:
            break
        prev_g = argM[j][g]
        i = j
        g = prev_g
        
    seq.reverse()
    return seq.tolist()
