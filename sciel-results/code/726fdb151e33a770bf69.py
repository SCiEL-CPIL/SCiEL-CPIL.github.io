import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [int(primes[0])]
    
    # dp[j][i] = length of longest valid sequence ending with primes[j], primes[i] (j < i)
    dp = np.zeros((N, N), dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    # Base case: all pairs (j, i) with j < i have length 2
    for j in range(N - 1):
        dp[j, j+1:] = 2
    
    global_max_len = 2
    best_j = 0
    best_i = 1
    
    for j in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        
        pj = primes[j]
        prev = dp[:j, j]  # dp[k][j] for k in 0..j-1
        
        if j == 0:
            continue
            
        # Compute suffix max of prev
        rev_prev = prev[::-1]
        cummax = np.maximum.accumulate(rev_prev)
        suf_max = cummax[::-1]  # suf_max[k] = max(prev[k:])
        
        # Compute suffix argmax (first occurrence of max in prev[k:])
        # For each k, we need the smallest index i >= k such that prev[i] == suf_max[k]
        # Use a scan from right to left, tracking the best (max value, smallest index)
        suf_arg = np.empty(j, dtype=np.int32)
        best_val = -1
        best_idx = -1
        for k in range(j - 1, -1, -1):
            if prev[k] > best_val:
                best_val = prev[k]
                best_idx = k
            # If prev[k] == best_val, we keep the existing best_idx (which is smaller since we're going right to left)
            # Actually, we want the FIRST occurrence, so if prev[k] == best_val, k is smaller, so update
            if prev[k] == best_val:
                best_idx = k
            suf_arg[k] = best_idx
        
        if j + 1 >= N:
            continue
            
        targets = primes[j+1:]
        thresholds = 2 * pj - targets
        Ks = np.searchsorted(primes, thresholds, side='left')
        
        valid_mask = Ks < j
        if not np.any(valid_mask):
            continue
            
        valid_indices = np.where(valid_mask)[0]
        i_indices = j + 1 + valid_indices
        k_indices = Ks[valid_indices]
        
        vals = suf_max[k_indices] + 1
        args = suf_arg[k_indices]
        
        dp[j, i_indices] = vals
        parent[j, i_indices] = args
        
        max_val = np.max(vals)
        if max_val > global_max_len:
            global_max_len = max_val
            idx_in_valid = np.argmax(vals)
            best_j = j
            best_i = int(i_indices[idx_in_valid])
    
    if global_max_len <= 1:
        return [int(primes[0])]
        
    seq = []
    curr_j = best_j
    curr_i = best_i
    
    seq.append(int(primes[curr_i]))
    seq.append(int(primes[curr_j]))
    
    k = parent[curr_j][curr_i]
    while k != -1:
        seq.append(int(primes[k]))
        curr_i = curr_j
        curr_j = k
        k = parent[curr_j][curr_i]
        
    seq.reverse()
    return seq
