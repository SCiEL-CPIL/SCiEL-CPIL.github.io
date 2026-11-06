import time
from math import isqrt
import numpy as np

def construct():
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Generate primes up to 10000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    P = get_primes(10000)
    P_arr = np.array(P, dtype=np.int64)
    N = len(P)
    
    # dp[j][i] = length of longest valid subsequence ending with P[j], P[i] (j < i)
    dp = np.zeros((N, N), dtype=np.int32)
    par = np.full((N, N), -1, dtype=np.int32)
    
    max_len = 0
    best_j = -1
    best_i = -1
    
    # Iterate j from 0 to N-1
    for j in range(N):
        if time.time() > deadline:
            break
        if j == 0:
            continue
            
        # Prepare prefix max for column j (rows 0 to j-1)
        # pref_val[k] = max(dp[0..k][j]), pref_idx[k] = argmax index
        # We need max over k' in [0, cnt-1]
        pref_val = np.zeros(j, dtype=np.int32)
        pref_idx = np.full(j, -1, dtype=np.int32)
        
        # Initialize first element
        pref_val[0] = dp[0, j]
        pref_idx[0] = 0
        
        for k in range(1, j):
            if dp[k, j] > pref_val[k-1]:
                pref_val[k] = dp[k, j]
                pref_idx[k] = k
            else:
                pref_val[k] = pref_val[k-1]
                pref_idx[k] = pref_idx[k-1]
                
        # Vectorized inner loop for i from j+1 to N-1
        i_indices = np.arange(j+1, N)
        X = 2 * P_arr[j] - P_arr[i_indices]
        
        # For each i, find cnt = number of primes P[k] <= X for k < j
        # searchsorted with side='right' gives the index of the first element > X
        # which is exactly the count of elements <= X
        cnt_arr = np.searchsorted(P_arr[:j], X, side='right')
        
        # Where cnt > 0, we have a valid predecessor
        valid_mask = cnt_arr > 0
        
        # For valid entries, get the value and predecessor index
        valid_i_idx = np.where(valid_mask)[0]
        if len(valid_i_idx) > 0:
            # We need pref_val[cnt-1]
            # cnt_arr[valid_i_idx] contains the counts
            # Indices into pref_val are cnt-1
            pref_indices = cnt_arr[valid_i_idx] - 1
            vals = pref_val[pref_indices]
            k_bests = pref_idx[pref_indices]
            
            # Update dp and par for these entries
            actual_i = i_indices[valid_i_idx]
            dp[j, actual_i] = vals + 1
            par[j, actual_i] = k_bests
            
            # Track max
            max_vals = vals + 1
            local_max_idx = np.argmax(max_vals)
            local_max_val = max_vals[local_max_idx]
            if local_max_val > max_len:
                max_len = int(local_max_val)
                best_j = j
                best_i = int(actual_i[local_max_idx])
        
        # For invalid entries (cnt == 0), dp remains 0 initially, but should be 2
        # The pair (P[j], P[i]) itself is a valid sequence of length 2
        invalid_i_idx = np.where(~valid_mask)[0]
        if len(invalid_i_idx) > 0:
            actual_i_invalid = i_indices[invalid_i_idx]
            dp[j, actual_i_invalid] = 2
            # par stays -1
            
            # Check if 2 > max_len (only matters if max_len < 2)
            if max_len < 2:
                max_len = 2
                best_j = j
                best_i = int(actual_i_invalid[0])
        
        # Also check if any valid entries resulted in length 2?
        # vals are from pref_val, which are from dp[*, j].
        # dp[*, j] is at least 2 (base case set in previous iterations).
        # So vals >= 2, dp[j, i] >= 3.
        # So valid entries always improve over base case 2.
        # But if pref_val was 0? No, dp is initialized to 0, but set to 2 in base cases.
        # Wait, dp[0, j] is set when j was the inner index in the j=0 iteration?
        # No, j=0 is skipped.
        # When j=1, dp[0, 1] is set.
        # So for j>0, dp[0..j-1, j] are all non-zero (at least 2).
    
    # Reconstruct sequence
    if max_len == 0:
        return []
        
    if max_len == 2:
        return [int(P[best_j]), int(P[best_i])]
        
    seq = [int(P[best_i]), int(P[best_j])]
    j_cur = best_j
    i_cur = best_i
    
    while par[j_cur, i_cur] != -1:
        k = int(par[j_cur, i_cur])
        seq.append(int(P[k]))
        i_cur = j_cur
        j_cur = k
        
    seq.reverse()
    return seq
