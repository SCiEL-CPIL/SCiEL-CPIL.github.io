import numpy as np
import time

def construct():
    deadline = time.time() + 55.0
    
    N_limit = 10000
    is_prime = np.ones(N_limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N_limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N < 2:
        return primes.tolist()
        
    dp = np.zeros((N, N), dtype=np.int32)
    next_p = np.full((N, N), -1, dtype=np.int32)
    
    for j in range(N-1, -1, -1):
        if time.time() > deadline:
            break
            
        # Precompute suffix max for dp[j, k] for k > j
        suf_max_val = np.zeros(N, dtype=np.int32)
        suf_max_idx = np.full(N, -1, dtype=np.int32)
        
        if N > j + 1:
            suf_max_val[N-1] = dp[j, N-1]
            if dp[j, N-1] > 0:
                suf_max_idx[N-1] = N-1
                
            for k in range(N-2, j, -1):
                if dp[j, k] > suf_max_val[k+1]:
                    suf_max_val[k] = dp[j, k]
                    suf_max_idx[k] = k
                else:
                    suf_max_val[k] = suf_max_val[k+1]
                    suf_max_idx[k] = suf_max_idx[k+1]
                    
        # Two-pointer for k_min
        k_ptr = j + 1
        
        for i in range(j-1, -1, -1):
            if time.time() > deadline:
                break
                
            gap = primes[j] - primes[i]
            target = primes[j] + gap
            
            # Advance k_ptr while primes[k_ptr] < target
            while k_ptr < N and primes[k_ptr] < target:
                k_ptr += 1
                
            if k_ptr < N and k_ptr > j:
                best_val = suf_max_val[k_ptr]
                if best_val > 0:
                    dp[i, j] = 1 + best_val
                    next_p[i, j] = suf_max_idx[k_ptr]
                else:
                    dp[i, j] = 2
            else:
                dp[i, j] = 2
                
    max_len = 0
    best_i = 0
    best_j = 1
    for i in range(N):
        for j in range(i+1, N):
            if dp[i, j] > max_len:
                max_len = dp[i, j]
                best_i = i
                best_j = j
                
    if max_len == 0:
        return primes.tolist()
        
    seq = [primes[best_i], primes[best_j]]
    curr_i, curr_j = best_i, best_j
    while next_p[curr_i, curr_j] != -1:
        next_k = next_p[curr_i, curr_j]
        seq.append(primes[next_k])
        curr_i, curr_j = curr_j, next_k
        
    return seq
