"""Stronger algorithm for longest increasing prime subsequence with nondecreasing gaps."""
import time
import numpy as np

def _sieve(n):
    """Return a numpy boolean array where arr[i] is True iff i is prime."""
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    return is_prime

def _get_primes(is_prime, limit):
    """Extract list of primes up to limit from sieve."""
    return [int(p) for p in range(2, limit + 1) if is_prime[p]]

def _dp_longest(primes, gap_limit, deadline, start_time):
    """
    DP over primes: dp[i][g] = length of best nondecreasing-gap subsequence ending at primes[i]
    with last gap exactly g (or <=g depending on formulation).
    
    We'll use: for each prime i, and each possible last gap g, 
    we can extend a subsequence ending at prime j < i where primes[i]-primes[j] = g_new >= g_old.
    
    To be efficient, we note that gaps between primes up to 10000 are at most ~200.
    We'll store for each prime index i and gap g the max length.
    
    Alternative: dp[i] = best length ending at i, and prev_gap[i] = the gap used.
    But we need to track the last gap to enforce nondecreasing.
    
    Let's use: best[i][g] = max length of nondecreasing-gap subsequence ending at primes[i] 
    with last gap = g.
    
    Transition: for each j < i, let d = primes[i] - primes[j]. 
    Then for any g <= d, best[i][d] = max(best[i][d], best[j][g] + 1).
    
    This is O(n^2 * max_gap) which is too slow for n~1200.
    
    Better: For each prime i, we want to find for each possible new gap d = primes[i] - primes[j],
    the max over j with primes[i]-primes[j]=d of max over g<=d of best[j][g].
    
    We can precompute prefix max: for each j, prefix_best[j][g] = max(best[j][0], ..., best[j][g]).
    Then best[i][d] = 1 + max over j where primes[j] = primes[i]-d of prefix_best[j][d].
    
    But iterating over all j for each i and d is still expensive.
    
    Let's think differently. max_gap is small (~200). n is ~1200.
    For each i, for each d in 1..max_gap:
        j_target = primes[i] - d
        If j_target is a prime, find its index j, then best[i][d] = 1 + prefix_best[j][d]
    
    This is O(n * max_gap) = 1200 * 200 = 240,000 which is fine.
    """
    n = len(primes)
    if n == 0:
        return [], []
    
    # Find max possible gap between consecutive primes (or any two primes)
    # Gaps between any two primes <= 10000 can be up to 10000, but nondecreasing sequences
    # with large gaps will be short. Let's set gap_limit to something reasonable.
    # Actually, for the DP to be useful, we need to consider gaps that can appear.
    # Let's use gap_limit = 200 initially, but we may need more.
    # The maximum gap between consecutive primes below 10000 is about 114.
    # But we can have larger gaps in a subsequence. However, if gap is too large, 
    # the sequence will be short. Let's be safe and use gap_limit = 500.
    
    # Actually, let's compute the maximum gap we need.
    # For a subsequence with nondecreasing gaps, if the last gap is d, 
    # and we have k elements, the total span is at least d*(k-1)/2 (if gaps are 1,1,...,d).
    # With span <= 10000, k <= 20000/d + 1. For d=200, k <= 101. For d=500, k <= 41.
    # So larger gaps give shorter sequences. We'll use gap_limit = 300 to be safe.
    
    gap_limit = 300
    
    # best[i][g] = max length of nondecreasing-gap subsequence ending at primes[i] with last gap = g
    # We'll use a list of arrays for efficiency
    best = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    # Initialize: each prime alone is a subsequence of length 1 with "last gap" = 0 (or -1)
    for i in range(n):
        best[i][0] = 1  # gap 0 means no previous element
    
    # prefix_best[i][g] = max(best[i][0], ..., best[i][g])
    prefix_best = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    
    # Create a map from prime value to its index
    prime_to_idx = {primes[i]: i for i in range(n)}
    
    for i in range(1, n):
        # For each possible gap d
        for d in range(1, gap_limit + 1):
            prev_val = primes[i] - d
            if prev_val < 2 or prev_val not in prime_to_idx:
                continue
            j = prime_to_idx[prev_val]
            # We can extend a subsequence ending at j with last gap g <= d
            # The best such length is prefix_best[j][d]
            candidate = prefix_best[j][d] + 1
            if candidate > best[i][d]:
                best[i][d] = candidate
        
        # Compute prefix_best[i]
        for g in range(1, gap_limit + 1):
            prefix_best[i][g] = max(prefix_best[i][g-1], best[i][g])
        
        # Check deadline
        if (i % 100 == 0) and time.time() > deadline:
            break
    
    # Find the overall best
    max_len = 0
    best_i = -1
    best_g = -1
    for i in range(n):
        for g in range(1, gap_limit + 1):
            if best[i][g] > max_len:
                max_len = best[i][g]
                best_i = i
                best_g = g
        # Also check gap 0
        if best[i][0] > max_len:
            max_len = best[i][0]
            best_i = i
            best_g = 0
    
    if best_i < 0:
        return [primes[0]], [0]
    
    # Reconstruct the subsequence
    # We need to store parent pointers
    # This requires redoing the DP with parent tracking, or storing parents
    
    # For simplicity, let's just return the length for now and reconstruct separately
    # Actually, we need to return the actual primes.
    
    # Let's redo with parent tracking
    parent = [(-1, -1) for _ in range(n)]  # parent[i] = (j, g_prev) where j is previous index and g_prev is the gap used at j
    
    best2 = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    prefix_best2 = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    
    for i in range(n):
        best2[i][0] = 1
        prefix_best2[i][0] = 1
    
    for i in range(1, n):
        for d in range(1, gap_limit + 1):
            prev_val = primes[i] - d
            if prev_val < 2 or prev_val not in prime_to_idx:
                continue
            j = prime_to_idx[prev_val]
            candidate = prefix_best2[j][d] + 1
            if candidate > best2[i][d]:
                best2[i][d] = candidate
                # Find which g at j gave prefix_best2[j][d]
                # We need to find the g that achieved the max
                # This is O(gap_limit) per update, which is acceptable
                for g in range(d, -1, -1):
                    if best2[j][g] == prefix_best2[j][d]:
                        parent[i] = (j, g)
                        break
        for g in range(1, gap_limit + 1):
            prefix_best2[i][g] = max(prefix_best2[i][g-1], best2[i][g])
        
        if (i % 100 == 0) and time.time() > deadline:
            break
    
    # Find best
    max_len = 0
    best_i = -1
    best_g = -1
    for i in range(n):
        for g in range(1, gap_limit + 1):
            if best2[i][g] > max_len:
                max_len = best2[i][g]
                best_i = i
                best_g = g
        if best2[i][0] > max_len:
            max_len = best2[i][0]
            best_i = i
            best_g = 0
    
    if best_i < 0:
        return [primes[0]], [0]
    
    # Reconstruct
    result = []
    cur_i = best_i
    cur_g = best_g
    while cur_i >= 0:
        result.append(primes[cur_i])
        if cur_g == 0:
            break
        prev_i, prev_g = parent[cur_i]
        cur_i = prev_i
        cur_g = prev_g
    
    result.reverse()
    return result, [primes[i+1] - primes[i] for i in range(len(result)-1)] if len(result) > 1 else []

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    limit = 10000
    is_prime = _sieve(limit)
    primes = _get_primes(is_prime, limit)
    
    if not primes:
        return []
    
    # Try DP approach
    if time.time() < deadline:
        result, gaps = _dp_longest(primes, 300, deadline, start_time)
        if result and len(result) > 0:
            # Verify the result
            if all(result[i+1] - result[i] >= result[i] - result[i-1] for i in range(1, len(result)-1)):
                if all(2 <= p <= 10000 and is_prime[p] for p in result):
                    return result
    
    # Fallback: greedy
    values = [primes[0]]
    for p in primes[1:]:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    
    return values
