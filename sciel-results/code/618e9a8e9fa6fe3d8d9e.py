import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N <= 1:
        return primes
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    dp = np.full((N, N), 2, dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    best_len = 2
    best_end = (0, 1)
    
    # For each j, we need suffix_max[i][k] = max(dp[k][i], ..., dp[i-1][i]) for k < i
    # But we can compute this incrementally as j increases
    
    # Actually, let's use a different approach:
    # For fixed j, dp[i][j] = 1 + max(dp[k][i]) for all k < i where primes[i] - primes[k] <= primes[j] - primes[i]
    # i.e., primes[k] >= 2*primes[i] - primes[j]
    
    # The constraint is: gap[i->j] >= gap[k->i], i.e., primes[j]-primes[i] >= primes[i]-primes[k]
    # So primes[k] >= 2*primes[i] - primes[j]
    
    # For each i < j, find the smallest index k_min such that primes[k_min] >= 2*primes[i] - primes[j]
    # Then dp[i][j] = 1 + max(dp[k][i]) for k in [k_min, i-1]
    
    # We need to maintain for each i: a data structure to query max dp[k][i] for k in [k_min, i-1]
    # Since k_min only increases as j increases (for fixed i), we can use a segment tree or sparse table
    
    # Given N ~ 1229, a simple approach with precomputed suffix max works well
    
    # Precompute for each i, the suffix max of dp[:, i]
    # But dp[:, i] changes as we process j > i... 
    # Actually no: dp[k][i] is computed when we process j=i, and doesn't change after.
    # So for each i, once we finish processing j=i, dp[:, i] is fixed.
    
    # Let's restructure: process j from 1 to N-1
    # When processing j, dp[i][j] depends on dp[k][i] for k < i
    # dp[k][i] was computed when we processed j=i (if i > k)
    # So all dp[k][i] for k < i are already computed when we start processing j > i
    
    # For each i, after we finish all j > i processing... no wait.
    # dp[k][i] is computed when the outer loop is at j=i.
    # So when outer loop is at j, for all i < j, dp[k][i] for k < i are already computed.
    
    # So for each i, we can precompute suffix_max_i[k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # This is valid once we've processed j=i.
    
    # But we need this for i < j, and we compute it after processing j=i.
    # So when processing j, for all i < j, suffix_max_i is already available.
    
    # Let's precompute suffix_max for each i as we go
    
    suffix_max = [None] * N  # suffix_max[i][k] = max(dp[k][i], ..., dp[i-1][i])
    suffix_argmax = [None] * N  # index of the max
    
    for j in range(1, N):
        # First, compute suffix_max for i = j-1 (now that dp[:, j-1] is complete)
        i_prev = j - 1
        if i_prev > 0:
            col = dp[:i_prev, i_prev]  # dp[0][i_prev], ..., dp[i_prev-1][i_prev]
            # Compute suffix max
            sm = np.empty(i_prev, dtype=np.int32)
            sm[i_prev - 1] = col[i_prev - 1]
            argmax = np.empty(i_prev, dtype=np.int32)
            argmax[i_prev - 1] = i_prev - 1
            for k in range(i_prev - 2, -1, -1):
                if col[k] >= sm[k + 1]:
                    sm[k] = col[k]
                    argmax[k] = k
                else:
                    sm[k] = sm[k + 1]
                    argmax[k] = argmax[k + 1]
            suffix_max[i_prev] = sm
            suffix_argmax[i_prev] = argmax
        
        # Now compute dp[i][j] for all i < j
        pj = primes_arr[j]
        
        # For each i in [0, j-1]:
        #   threshold = 2 * primes[i] - primes[j]
        #   k_min = lower_bound(primes, threshold)  # first index where primes[k] >= threshold
        #   if k_min < i: dp[i][j] = 1 + suffix_max[i][k_min]
        
        # Vectorize: compute all thresholds
        thresholds = 2 * primes_arr[:j] - pj
        
        # Find k_min for each i
        k_min_arr = np.searchsorted(primes_arr, thresholds, side='left')
        
        # Only valid if k_min_arr[i] < i
        valid = k_min_arr < np.arange(j)
        
        if np.any(valid):
            idxs = np.where(valid)[0]
            km_vals = k_min_arr[idxs]
            
            # For each valid i, get suffix_max[i][km_vals[i]]
            # This requires a loop since suffix_max[i] has different lengths
            new_vals = np.empty(len(idxs), dtype=np.int32)
            new_parents = np.empty(len(idxs), dtype=np.int32)
            
            for t, ii in enumerate(idxs):
                km = km_vals[t]
                new_vals[t] = 1 + suffix_max[ii][km]
                new_parents[t] = suffix_argmax[ii][km]
            
            # Update dp and parent
            for t, ii in enumerate(idxs):
                if new_vals[t] > dp[ii, j]:
                    dp[ii, j] = new_vals[t]
                    parent[ii, j] = new_parents[t]
                    if new_vals[t] > best_len:
                        best_len = new_vals[t]
                        best_end = (ii, j)
        
        # Check time
        if j % 50 == 0 and time.time() > deadline:
            break
    
    # Reconstruct the sequence
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    while parent[ci, cj] != -1:
        ck = parent[ci, cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
    
    result.reverse()
    return result
