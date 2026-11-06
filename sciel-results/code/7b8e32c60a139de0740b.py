import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    deadline = time.time() + 55
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n < 2:
        return primes
    
    # dp[i][j] = longest valid subsequence ending with primes[i], primes[j] (i < j)
    # parent[i][j] = the index h < i that gives the best extension to (i,j)
    dp = np.full((n, n), 2, dtype=np.int32)
    parent = np.full((n, n), -1, dtype=np.int32)
    
    primes_np = np.array(primes, dtype=np.int64)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        # Get dp values and gaps for h < i
        dp_col = dp[:i, i]
        gaps_h = primes_np[i] - primes_np[:i]
        
        # Sort gaps_h and keep track of original indices
        sorted_indices = np.argsort(gaps_h)
        g_h_sorted = gaps_h[sorted_indices]
        d_h_sorted = dp_col[sorted_indices]
        
        # Compute prefix max of d_h_sorted
        pmax = np.maximum.accumulate(d_h_sorted)
        
        # Compute prefix argmax (index in sorted array)
        # This is a simple loop, O(i)
        pmax_arg = np.zeros(len(d_h_sorted), dtype=np.int32)
        if len(d_h_sorted) > 0:
            pmax_arg[0] = 0
            for k in range(1, len(d_h_sorted)):
                if d_h_sorted[k] > pmax[k-1]:
                    pmax_arg[k] = k
                else:
                    pmax_arg[k] = pmax_arg[k-1]
        
        # Get gaps for j > i
        gaps_j = primes_np[i+1:] - primes_np[i]
        
        if len(gaps_j) == 0:
            continue
            
        # Find rightmost index where g_h_sorted <= gaps_j
        # searchsorted with side='right' gives the index where gap_j would be inserted
        # to maintain sorted order, i.e., the count of elements <= gap_j
        k = np.searchsorted(g_h_sorted, gaps_j, side='right')
        
        # valid entries are those with k > 0
        valid_mask = k > 0
        if not np.any(valid_mask):
            continue
            
        k_valid = k[valid_mask] - 1
        best_vals = pmax[k_valid]
        best_args = pmax_arg[k_valid]
        
        # Map back to original h indices
        parent_indices = sorted_indices[best_args]
        
        # Update dp and parent
        # j indices are i+1 + np.where(valid_mask)[0]
        j_indices = np.where(valid_mask)[0] + i + 1
        
        dp[i, j_indices] = best_vals + 1
        parent[i, j_indices] = parent_indices
        
    # Find the best pair (i, j) maximizing dp[i][j]
    # We only care about i < j
    # Create a mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dp_masked = np.where(mask, dp, 0)
    
    max_val = np.max(dp_masked)
    if max_val <= 2:
        # Find first pair
        idx = np.argmax(dp_masked)
        best_i, best_j = np.unravel_index(idx, (n, n))
        return [primes[best_i], primes[best_j]]
        
    # Find indices where dp == max_val
    # We want the one that allows reconstruction, but any max is fine
    # Let's just pick the first one
    max_indices = np.argwhere(dp_masked == max_val)
    best_i, best_j = max_indices[0]
    
    # Reconstruct the sequence
    seq = []
    i, j = best_i, best_j
    while True:
        seq.append(primes[j])
        h = parent[i, j]
        if h == -1:
            seq.append(primes[i])
            break
        j = i
        i = h
    seq.reverse()
    
    return seq
