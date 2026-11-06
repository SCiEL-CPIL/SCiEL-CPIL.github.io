"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    deadline = time.time() + 50  # safety margin before 60s kill
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n < 2:
        return primes
    
    # dp[i][j] = longest valid subsequence ending with primes[i], primes[j] (i < j)
    # We want max dp[i][j] over all pairs, then reconstruct.
    
    # For memory efficiency with n~1229, use a 2D list of shorts
    # dp[i][j] for i < j
    # Initialize: dp[i][j] = 2 for all i < j
    
    # Transition: dp[i][j] = max(2, 1 + max(dp[h][i] for h < i if primes[i]-primes[h] <= primes[j]-primes[i]))
    
    # For each i, as we iterate j > i, the gap primes[j]-primes[i] increases.
    # For fixed i, we need: for each j, max over h<i of dp[h][i] where primes[i]-primes[h] <= primes[j]-primes[i]
    
    # Precompute for each i: list of (gap_from_h, dp[h][i]) for h < i
    # Then for each j, binary search or just iterate since gaps are monotonic in j.
    
    # Actually, let's use a different approach:
    # For each i (0 to n-1), maintain best[i][gap_index] where gap_index corresponds to primes[i]-primes[h]
    # But gaps can be up to 10000, so we can use an array of size 10001.
    
    # For fixed i:
    #   For each h < i: gap = primes[i] - primes[h], candidate = dp[h][i]
    #   We want prefix max: for a given max_gap G, best = max of dp[h][i] over h<i where primes[i]-primes[h] <= G
    #   As j increases, G = primes[j] - primes[i] increases, so we can process incrementally.
    
    # Let's use: for each i, build an array best_for_gap[g] = max dp[h][i] for h<i with primes[i]-primes[h] <= g
    # But that's O(n * 10000) which is ~12M, fine.
    
    # Actually simpler: for each i, collect (primes[i]-primes[h], dp[h][i]) for h < i, sort by gap,
    # then compute prefix max. Then for each j > i, binary search for the right gap.
    
    # dp stored as list of lists: dp[i][j] only for i < j
    # Use a flat approach or list of arrays
    
    # Let's use: dp = [[0]*n for _ in range(n)] but only fill upper triangle
    # n = 1229, so n^2 = ~1.5M entries, each a small int. Fine.
    
    dp = [[0] * n for _ in range(n)]
    
    # Base case: sequences of length 2
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 2
    
    # For each middle index i, compute transitions
    # For fixed i, we want to update dp[i][j] for all j > i
    # dp[i][j] = max(dp[i][j], 1 + max(dp[h][i] for h < i where primes[i]-primes[h] <= primes[j]-primes[i]))
    
    # Process i from 1 to n-1 (i=0 has no h < 0)
    for i in range(1, n):
        if time.time() > deadline:
            break
        # Collect (gap, dp[h][i]) for h < i
        pairs = []
        for h in range(i):
            gap = primes[i] - primes[h]
            pairs.append((gap, dp[h][i]))
        # Sort by gap
        pairs.sort()
        # Compute prefix max
        prefix_max = []
        cur_max = 0
        for gap, val in pairs:
            if val > cur_max:
                cur_max = val
            prefix_max.append(cur_max)
        
        # For each j > i, find max gap allowed = primes[j] - primes[i]
        # Binary search in sorted gaps
        import bisect
        gaps_sorted = [p[0] for p in pairs]
        
        for j in range(i + 1, n):
            max_gap = primes[j] - primes[i]
            # Find rightmost index where gap <= max_gap
            idx = bisect.bisect_right(gaps_sorted, max_gap) - 1
            if idx >= 0:
                candidate = 1 + prefix_max[idx]
                if candidate > dp[i][j]:
                    dp[i][j] = candidate
    
    # Find the best pair (i, j) maximizing dp[i][j]
    best_len = 2
    best_i = 0
    best_j = 1
    for i in range(n):
        for j in range(i + 1, n):
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
    
    if best_len <= 2:
        return [primes[best_i], primes[best_j]]
    
    # Reconstruct the sequence
    # We need to trace back. dp[i][j] tells us the length ending at (i,j).
    # To reconstruct, we need to know which h was used. Let's store parent pointers.
    
    # Actually, let's redo with parent tracking.
    # This is getting complex. Let me use a different reconstruction approach.
    
    # Alternative: store for each (i,j) the best predecessor h.
    # Let me redo the DP with parent tracking.
    
    dp2 = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dp2[i][j] = 2
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        pairs = []
        for h in range(i):
            gap = primes[i] - primes[h]
            pairs.append((gap, h, dp2[h][i]))
        pairs.sort()
        prefix_max = []
        prefix_arg = []
        cur_max = 0
        cur_arg = -1
        for gap, h, val in pairs:
            if val > cur_max:
                cur_max = val
                cur_arg = h
            prefix_max.append(cur_max)
            prefix_arg.append(cur_arg)
        
        import bisect
        gaps_sorted = [p[0] for p in pairs]
        
        for j in range(i + 1, n):
            max_gap = primes[j] - primes[i]
            idx = bisect.bisect_right(gaps_sorted, max_gap) - 1
            if idx >= 0:
                candidate = 1 + prefix_max[idx]
                if candidate > dp2[i][j]:
                    dp2[i][j] = candidate
                    parent[i][j] = prefix_arg[idx]
    
    best_len = 2
    best_i = 0
    best_j = 1
    for i in range(n):
        for j in range(i + 1, n):
            if dp2[i][j] > best_len:
                best_len = dp2[i][j]
                best_i = i
                best_j = j
    
    if best_len <= 2:
        return [primes[best_i], primes[best_j]]
    
    # Reconstruct
    seq = []
    i, j = best_i, best_j
    while j != -1:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            break
        j = i
        i = h
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
