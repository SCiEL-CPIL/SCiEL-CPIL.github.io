"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # We use a list of lists. dp[j] is a list of length j, where dp[j][i] is the value.
    # Or a full N x N matrix. N=1229, so 1.5M entries.
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # suf_max_col[j] will store suffix maxima for column j.
    # suf_max_col[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j])
    # We compute this after column j is fully populated.
    suf_max_col = [None] * N
    
    # Column 0 is empty (no i < 0)
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    for j in range(1, N):
        # Check time
        if time.time() > deadline:
            break
            
        # Compute dp[i][j] for all i < j
        # We need suf_max_col[i] for i < j. These are already computed.
        
        # Optimization: Precompute suffix maxima for column j is done AFTER this loop.
        # But we need suf_max_col[i] for i < j.
        
        for i in range(0, j):
            gap = primes[j] - primes[i]
            # Condition: primes[i] - primes[k] <= gap
            # primes[k] >= primes[i] - gap
            threshold = primes[i] - gap
            
            # Find smallest k < i such that primes[k] >= threshold
            # Since primes is sorted, we can binary search or use a pointer.
            # For fixed i, as j increases, gap increases, threshold decreases.
            # But here i is fixed, j varies in the outer loop? No, j is outer, i is inner.
            # So for fixed j, i varies.
            
            # We need max dp[k][i] for k in [L, i-1] where primes[k] >= threshold.
            # This is max over a suffix of the segment dp[0..i-1][i].
            # We have suf_max_col[i] available.
            
            # Find L: smallest index in [0, i-1] such that primes[L] >= threshold.
            # If no such index, L = i (empty range).
            
            # Binary search in primes[0:i] for threshold.
            # We want the first index L such that primes[L] >= threshold.
            lo = 0
            hi = i
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    hi = mid
                else:
                    lo = mid + 1
            L = lo
            
            if L < i:
                # Max is suf_max_col[i][L]
                val = suf_max_col[i][L]
                if val > 0:
                    dp[i][j] = val + 1
                    # Find which k gave this max
                    # We need to find k in [L, i-1] with dp[k][i] == val
                    # We can just scan? i is up to 1229. Scanning is O(i).
                    # Total O(N^3) in worst case?
                    # The binary search is O(log N). The scan is O(N).
                    # Inner loop is O(N). Outer is O(N). Total O(N^3)?
                    # Wait, the scan is only if val > 0.
                    # To avoid O(N) scan, we could store argmax in suf_max_col.
                    # Let's modify suf_max_col to store (value, index).
                    
                    # For now, let's just find the index.
                    # Actually, we can precompute argmax as well.
                    # Let's stick to simple scanning for now, N is small enough?
                    # 1229^3 is too big. 1229^2 * 10 is ok.
                    # The scan is over [L, i-1]. Length is i-L.
                    # On average i/2. So O(N^3/2). Might be slow in Python.
                    
                    # Let's optimize by storing argmax in suf_max_col.
                    # We'll do that in the next block.
                    
                    # For now, find k
                    best_k = -1
                    for k in range(L, i):
                        if dp[k][i] == val:
                            best_k = k
                            break
                    prev[i][j] = best_k
                else:
                    dp[i][j] = 2
            else:
                dp[i][j] = 2
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # After processing all i for this j, compute suf_max_col[j]
        # suf_max_col[j][k] = max(dp[k][j], dp[k+1][j], ...) for k in [0, j-1]
        # We need this for future j' > j where i=j.
        # So we need suf_max_col[j] to be ready when i=j is used as the 'i' in the inner loop.
        
        # Initialize suf_max_col[j]
        sm = [0] * j
        if j > 0:
            sm[j-1] = dp[j-1][j]
            for k in range(j-2, -1, -1):
                if dp[k][j] > sm[k+1]:
                    sm[k] = dp[k][j]
                else:
                    sm[k] = sm[k+1]
        suf_max_col[j] = sm
        
    # Reconstruct sequence
    # Start from best_j, best_i
    # The sequence ends with primes[best_i], primes[best_j]
    # prev[best_i][best_j] is the index k such that the sequence ended with primes[k], primes[best_i]
    
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    while curr_i != -1:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
