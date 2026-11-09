import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 75.0
    
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0].astype(np.int32)
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
        if j % 100 == 0:
            remaining = DEADLINE - (time.time() - start_time)
            if remaining < 5.0:
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
        # Vectorized: for each k, find smallest index i >= k where prev[i] == suf_max[k]
        # Use a trick: create indices, broadcast, and find first match
        # Alternative: use argsort-based approach
        # For each k, we want argmin of (index) subject to prev[index] == suf_max[k] and index >= k
        
        # Efficient approach: process from right to left with numpy
        # Create a 2D comparison: for each k, compare prev[k:] with suf_max[k]
        # This is O(j^2) memory which is too much for large j
        
        # Better: use a cumulative approach
        # For each position k, the answer is the smallest index >= k where prev equals suf_max[k]
        # We can compute this by: for each value v, find all positions where prev==v,
        # then for each k, find the first position >= k where prev[k] == suf_max[k]
        
        # Actually, let's use a different vectorized approach:
        # Create an array where for each k, we mark positions >= k
        # Then find first position where prev[pos] == suf_max[k]
        
        # Simple but effective: use a loop but with numpy inner operations
        # Or: use the fact that we can compute this with a single pass
        # Let's use a trick with cumulative min of (index where value matches)
        
        # For each unique value, we can precompute the sorted list of indices
        # Then for each k, binary search for the first index >= k
        
        # But this is complex. Let's keep the loop but make it faster:
        suf_arg = np.empty(j, dtype=np.int32)
        best_val = -1
        best_idx = -1
        for k in range(j - 1, -1, -1):
            if prev[k] >= best_val:
                best_val = prev[k]
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
    
    # Verify and fix if needed
    if len(seq) < 2:
        return [int(primes[0])]
    
    # Check validity: primes, increasing, gaps non-decreasing
    gaps = np.diff(seq)
    if np.any(gaps < 0):
        return [int(primes[0])]
    if np.any(np.diff(gaps) < 0):
        # Gaps not non-decreasing, try to fix by keeping prefix
        for i in range(len(gaps) - 1, 0, -1):
            if np.all(np.diff(gaps[:i]) >= 0):
                seq = seq[:i+1]
                break
        else:
            return [int(primes[0])]
    
    return seq
