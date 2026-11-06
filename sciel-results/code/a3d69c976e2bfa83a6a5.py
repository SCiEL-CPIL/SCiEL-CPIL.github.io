# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # 1. Sieve of Eratosthenes for primes up to 10000
    LIMIT = 10000
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(LIMIT) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, LIMIT + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP: dp[i][j] = max length of valid subsequence ending at primes[i],
    #    with previous element being primes[j] (j < i)
    #    dp[i][j] = 1 + max{dp[j][k] : k < j and primes[j]-primes[k] <= primes[i]-primes[j]}
    #    Base: dp[i][j] = 1 if no valid k exists (subsequence is just [primes[j], primes[i]])
    #    Actually, a subsequence of length 1 has no gaps, so we can start anywhere.
    #    Let's define: a subsequence of length >= 2 ending at i with prev j.
    #    dp[i][j] = 2 + max{dp[j][k] - 1 : ...} hmm, let me redefine.
    
    # Let dp[i][j] = length of longest valid prime subsequence ending with (primes[j], primes[i])
    # i.e., the last two elements are primes[j] and primes[i], j < i.
    # Then dp[i][j] = 1 + max{ dp[j][k] : k < j and primes[j] - primes[k] <= primes[i] - primes[j] }
    # If no such k exists, dp[i][j] = 2 (just the two primes).
    
    # We also need to track the full path to reconstruct.
    
    # For efficiency: for each j, precompute for all possible gap thresholds the max dp[j][k].
    # For fixed j, the relevant k values are 0..j-1.
    # gap_k = primes[j] - primes[k] for k in 0..j-1
    # We want, for a query gap G, max{dp[j][k] : gap_k <= G}
    
    # dp is a 2D list, but we only need dp[j][k] for k < j.
    # Store dp as a list of lists: dp[i] is a dict or list mapping j -> value.
    # Since j < i, we can use a list of length i for dp[i].
    
    # parent[i][j] = k, the previous element before j in the subsequence ending at (j, i)
    
    dp = [None] * N  # dp[i] will be a list of length i, dp[i][j] for j in 0..i-1
    parent = [None] * N  # parent[i][j] = k or -1
    
    # Process i from 1 to N-1
    for i in range(1, N):
        if time.time() > deadline:
            break
        dp[i] = [2] * i  # default: just two primes
        parent[i] = [-1] * i
        
        # For each j in 0..i-1, we want to find max dp[j][k] for k < j with primes[j]-primes[k] <= primes[i]-primes[j]
        # But we need to process j in order. For a fixed i, we iterate j from 0 to i-1.
        # For each j, we need the prefix-max structure for dp[j][*].
        
        # Precompute for current i: for each j, the max dp[j][k] with gap <= primes[i]-primes[j]
        # This is expensive if done naively. Let's do it per j.
        
        for j in range(i):
            if time.time() > deadline:
                break
            gap_needed = primes[i] - primes[j]
            # Find max dp[j][k] for k in 0..j-1 with primes[j] - primes[k] <= gap_needed
            best = 2  # minimum: just (primes[j], primes[i])
            best_k = -1
            if j >= 1:
                # dp[j] is a list of length j
                # We need max dp[j][k] where primes[j] - primes[k] <= gap_needed
                # Iterate k from 0 to j-1
                # Optimization: precompute sorted by gap for each j? That's O(N^2 log N) total.
                # For now, linear scan. N=1229, so inner loop is at most 1228.
                # Total ops: sum_{i=1}^{N-1} sum_{j=0}^{i-1} j = sum_{i} i(i-1)/2 ≈ N^3/6 ≈ 300M for N=1229.
                # That's too slow in Python.
                
                # Better: for each j, precompute a list of (gap, dp_val, k) sorted by gap,
                # then prefix max. Then binary search for gap_needed.
                # But we're building dp incrementally... dp[j][k] is known when we process i > j.
                
                # Alternative: for each j, maintain a data structure as we add new dp[j][k] values.
                # But dp[j][k] is finalized when we finish processing row j, i.e., after all i > j are processed.
                # Wait no: dp[j][k] is computed when we process row j, which happens before row i (i > j).
                # So when we're at row i, all dp[j][*] for j < i are already computed.
                
                # So we can precompute for each j (after row j is done) a sorted list + prefix max.
                # Let's do that.
                
                pass  # We'll handle this with precomputation below
        
        # This approach is getting complex. Let me restructure.
    
    # Restructured approach:
    # After computing dp[j][*] for row j, build a helper structure for j:
    #   For j, we have pairs (gap_k, dp_val_k) for k in 0..j-1, where gap_k = primes[j]-primes[k]
    #   Sort by gap_k, compute prefix max of dp_val_k.
    #   Store as (sorted_gaps, prefix_max_dps, sorted_ks)
    
    # Let me redo this properly.
    
    dp = [None] * N
    parent = [None] * N
    # helper[j] = (gaps_sorted, prefix_max, ks_sorted) or None if j==0
    helper = [None] * N
    
    for j in range(N):
        if time.time() > deadline:
            break
        if j == 0:
            dp[0] = []
            parent[0] = []
            helper[0] = None
            continue
        
        dp[j] = [2] * j
        parent[j] = [-1] * j
        
        for k in range(j):
            gap_prev = primes[j] - primes[k]
            # Find max dp[k][m] for m < k with primes[k] - primes[m] <= gap_prev
            best = 2
            best_m = -1
            if k >= 1 and helper[k] is not None:
                gaps_s, prefix_max, ks_s = helper[k]
                # Binary search: find rightmost index where gaps_s[idx] <= gap_prev
                # Use bisect_right
                import bisect
                idx = bisect.bisect_right(gaps_s, gap_prev) - 1
                if idx >= 0:
                    val = prefix_max[idx]
                    if val > best:
                        best = val
                        best_m = ks_s[idx]
            if best > 2:
                dp[j][k] = best
                parent[j][k] = best_m
            else:
                dp[j][k] = 2
                parent[j][k] = -1
        
        # Build helper for j
        if j >= 1:
            pairs = []
            for k in range(j):
                g = primes[j] - primes[k]
                pairs.append((g, dp[j][k], k))
            pairs.sort(key=lambda x: x[0])
            gaps_s = [p[0] for p in pairs]
            prefix_max = [0] * len(pairs)
            running_max = 0
            for idx in range(len(pairs)):
                if pairs[idx][1] > running_max:
                    running_max = pairs[idx][1]
                prefix_max[idx] = running_max
            ks_s = [p[2] for p in pairs]
            helper[j] = (gaps_s, prefix_max, ks_s)
    
    # Find the best (i, j) pair
    if time.time() > deadline:
        # Return best found so far
        pass
    
    best_len = 1
    best_i = 0
    best_j = -1
    for i in range(N):
        if dp[i] is None:
            continue
        for j in range(i):
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
    
    if best_len <= 1:
        return [primes[0]]
    
    # Reconstruct
    result = []
    i = best_i
    j = best_j
    while j != -1:
        result.append(primes[i])
        result.append(primes[j])
        i = j
        j = parent[i][j] if i > 0 and j < i else -1
        # Wait, this is wrong. Let me think again.
        # dp[i][j] means subsequence ends with (primes[j], primes[i]).
        # parent[i][j] = k means the element before j is primes[k].
        # So the subsequence is: ..., primes[k], primes[j], primes[i]
        # To continue: set i = j, j = k, and look up parent[j][k]
    
    # Let me redo reconstruction properly
    result = []
    i_cur = best_i
    j_cur = best_j
    while j_cur != -1:
        result.append(primes[i_cur])
        result.append(primes[j_cur])
        next_j = parent[i_cur][j_cur]
        i_cur = j_cur
        j_cur = next_j
    
    result.reverse()
    return result

# EVOLVE-BLOCK-END
