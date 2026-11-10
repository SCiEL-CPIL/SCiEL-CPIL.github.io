"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    L = len(primes)
    
    if L <= 2:
        return primes
    
    # 2. DP Setup
    # dp[i][j] = max length of valid sequence ending at primes[i] with previous prime primes[j]
    # parent[i][j] = index k of the prime before primes[j] in the optimal sequence for dp[i][j]
    dp = [[0] * L for _ in range(L)]
    parent = [[-1] * L for _ in range(L)]
    
    # best_suffix_val[j][k] = max(dp[j][m] for m in [k, j-1])
    # best_suffix_idx[j][k] = argmax index m
    # We only need these for j < i.
    # We can compute best_suffix for row j after row j is fully processed.
    # But row j is processed when i=j.
    # However, dp[i][j] requires best_suffix[j].
    # So we need best_suffix[j] to be ready before we process i > j.
    # Since we iterate i from 0 to L-1, when we are at i, all j < i have their rows dp[j][*] computed.
    # So we can precompute best_suffix[j] right after finishing the inner loop for a specific i?
    # No, dp[i][j] is in row i.
    # The row j is dp[j][*].
    # We compute dp[i][j] for fixed i, varying j.
    # We need best_suffix[j].
    # So best_suffix[j] must be available.
    # best_suffix[j] depends on dp[j][*].
    # dp[j][*] is computed when the outer loop variable is j.
    # So, we should compute best_suffix[j] at the end of the outer loop iteration for j.
    
    best_suffix_val = [None] * L
    best_suffix_idx = [None] * L
    
    # Initialize for i=0, no j < 0, so loop does nothing.
    # best_suffix for row 0? Row 0 has no previous primes. dp[0][*] is empty/0.
    # We only use best_suffix[j] for j < i.
    
    for i in range(L):
        # Compute dp[i][j] for all j < i
        for j in range(i):
            # Threshold T = 2*primes[j] - primes[i]
            T = 2 * primes[j] - primes[i]
            
            # Find smallest k in [0, j-1] such that primes[k] >= T
            # primes is sorted.
            # We can use bisect_left
            import bisect
            k_min = bisect.bisect_left(primes, T, 0, j)
            
            if k_min < j:
                # We have valid k's in [k_min, j-1]
                # Get best value and index from best_suffix[j]
                val = best_suffix_val[j][k_min]
                idx = best_suffix_idx[j][k_min]
                dp[i][j] = val + 1
                parent[i][j] = idx
            else:
                # No valid k, sequence length is 2
                dp[i][j] = 2
                parent[i][j] = -1
        
        # After finishing row i (dp[i][*] for j < i is computed),
        # we can compute best_suffix for row i?
        # No, best_suffix[i] is needed for future rows where i is the 'j' index.
        # So yes, compute best_suffix[i] now.
        # best_suffix[i] is an array of size i.
        # best_suffix[i][k] = max(dp[i][m] for m in [k, i-1])
        # Note: dp[i][m] is only defined for m < i.
        
        if i > 0:
            bs_val = [0] * i
            bs_idx = [0] * i
            
            # Start from the end
            # m ranges from i-1 down to 0
            # bs_val[i-1] corresponds to m=i-1
            bs_val[i-1] = dp[i][i-1]
            bs_idx[i-1] = i-1
            
            for k in range(i-2, -1, -1):
                if dp[i][k] > bs_val[k+1]:
                    bs_val[k] = dp[i][k]
                    bs_idx[k] = k
                else:
                    bs_val[k] = bs_val[k+1]
                    bs_idx[k] = bs_idx[k+1]
            
            best_suffix_val[i] = bs_val
            best_suffix_idx[i] = bs_idx
            
        # Check deadline
        if time.time() > deadline:
            break

    # 3. Find the maximum length and reconstruct
    max_len = 0
    end_i = -1
    prev_j = -1
    
    for i in range(L):
        for j in range(i):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                prev_j = j
                
    if max_len == 0:
        return primes[:2] # Fallback
        
    # Reconstruct sequence
    seq = []
    curr = end_i
    prev = prev_j
    
    seq.append(primes[curr])
    seq.append(primes[prev])
    
    # Backtrack
    # State is (curr, prev).
    # The previous state was (prev, k) where k = parent[curr][prev]
    while prev != -1:
        k = parent[curr][prev]
        if k == -1:
            break
        seq.append(primes[k])
        curr = prev
        prev = k
        
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
