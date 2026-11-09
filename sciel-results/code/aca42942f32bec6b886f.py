"""Rewritten program using vectorized DP on a DAG of prime subsequences with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate all primes <= 10000
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j] (i < j)
    dp = np.zeros((n, n), dtype=np.int32)
    
    # Iterate i from 1 to n-1
    # For each i, we have already computed dp[h][i] for all h < i in previous iterations.
    # We compute suffix max of column i.
    # Then we update dp[i][j] for all j > i using vectorized operations.
    
    for i in range(1, n):
        # Get the column dp[:i, i]
        col = dp[:i, i]
        
        # Compute suffix maximum
        # S[k] = max(col[k:])
        # We can use np.maximum.accumulate on the reversed array and reverse back
        if i > 0:
            rev_col = col[::-1]
            suf_max_rev = np.maximum.accumulate(rev_col)
            S = suf_max_rev[::-1]
        else:
            S = np.array([], dtype=np.int32)
            
        # Now compute dp[i, j] for j in range(i+1, n)
        # j_indices = np.arange(i+1, n)
        # T_j = 2 * primes[i] - primes[j]
        # h_min_j = np.searchsorted(primes, T_j, side='left')
        
        # We only care about j where h_min_j < i
        # Let's compute for all j > i first
        
        j_indices = np.arange(i+1, n)
        if len(j_indices) == 0:
            continue
            
        T_j = 2 * primes[i] - primes[j_indices]
        h_min = np.searchsorted(primes, T_j, side='left')
        
        # Initialize dp[i, j_indices] to 2
        dp[i, j_indices] = 2
        
        # Mask for valid h_min (h_min < i)
        valid_mask = h_min < i
        
        if np.any(valid_mask):
            # For valid indices, compute candidate length
            # We need S[h_min]
            # Note: S is defined for indices 0 to i-1
            # h_min values are in [0, i-1] for valid_mask
            h_min_valid = h_min[valid_mask]
            j_valid = j_indices[valid_mask]
            
            # Get S values
            S_vals = S[h_min_valid]
            
            # Candidate length is 1 + S_vals
            # Since S_vals >= 2 (dp values are at least 2), candidate >= 3
            # So max(2, candidate) is just candidate
            candidate = 1 + S_vals
            
            # Update dp
            dp[i, j_valid] = candidate
    
    # Find the maximum length
    max_len = int(np.max(dp))
    
    # Reconstruct
    # Find (i, j) with max value
    i_star, j_star = np.unravel_index(np.argmax(dp), dp.shape)
    i_star, j_star = int(i_star), int(j_star)
    
    seq = [int(primes[j_star]), int(primes[i_star])]
    i, j = i_star, j_star
    
    while len(seq) < max_len:
        gap = primes[j] - primes[i]
        # Find h < i such that dp[h][i] + 1 == dp[i][j] and primes[i] - primes[h] <= gap
        found = False
        # Iterate h from i-1 down to 0
        for h in range(i-1, -1, -1):
            if dp[h][i] + 1 == dp[i][j] and (primes[i] - primes[h]) <= gap:
                seq.append(int(primes[h]))
                i, j = h, i
                found = True
                break
        if not found:
            # Fallback: should not happen if DP is correct
            break
            
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
