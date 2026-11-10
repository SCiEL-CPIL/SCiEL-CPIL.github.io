"""Optimal DP for longest nondecreasing-gap prime subsequence <=10000."""
import time
import bisect

def _sieve_primes(limit):
    """Return list of all primes <= limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    primes = _sieve_primes(10000)
    n = len(primes)

    if n < 2:
        return primes[:]

    # Precompute all pairwise gaps and build a gap-to-index mapping
    # gap[i][j] = primes[i] - primes[j] for j < i
    # We'll use a Fenwick tree over the gap dimension for O(log N) prefix max queries

    # For each i, we need to query: among all k < i with (primes[i] - primes[k]) <= gap_threshold,
    # find max dp[k][i] and the argmax k.
    
    # Strategy: Process primes in order. For each prime index i, we maintain a Fenwick tree
    # indexed by gap value. When we finish processing index i, we insert all (gap, dp, idx) 
    # tuples for pairs (k, i) where k < i.
    
    # Actually, let's think differently. The DP is:
    # dp[j][i] = 1 + max{dp[k][j] : k < j and primes[j]-primes[k] <= primes[i]-primes[j]}
    #           = 2 if no such k exists
    
    # This is a 2D problem. For fixed i, we need for each j < i:
    #   query: max dp[k][j] over k < j with (primes[j]-primes[k]) <= (primes[i]-primes[j])
    
    # Let's use the original approach but optimize the sorted_struct building.
    # The key insight: instead of building separate sorted lists per j, we can
    # precompute all gaps and use a single sorted structure.
    
    # Alternative: Use the fact that gaps are bounded by 10000.
    # Max gap = primes[-1] - primes[0] <= 10000
    # We can use a Fenwick tree of size ~10000 for each "layer" j.
    
    # But that's still O(N) Fenwick trees. Let's try a different approach.
    
    # For each j, we need to answer queries of the form:
    #   "What is the max dp[k][j] for k < j with primes[j]-primes[k] <= G?"
    # where G varies per i.
    
    # We can precompute for each j: sort all (primes[j]-primes[k], dp[k][j], k) by gap.
    # Then for a query with threshold G, binary search to find the rightmost element
    # with gap <= G, and use prefix max.
    
    # This is what the original does. Let's optimize it.
    
    # Optimization: Instead of storing full lists, use arrays and precompute prefix maxes.
    # Also, we can avoid rebuilding the sorted structure from scratch each time.
    
    # Let's try: for each j, precompute the sorted gaps and prefix maxes once.
    
    dp_table = [None] * n
    argmax_table = [None] * n
    sorted_struct = [None] * n  # (sorted_gaps, prefix_maxes, prefix_argmax)

    best_length = 2
    best_i = 1
    best_j = 0

    # Precompute: for each pair (j, i) with j < i, the gap
    # And for each j, collect all (gap, k) pairs where k < j
    
    # Actually, let's just do the original but with better memory layout.
    
    for i in range(1, n):
        # Compute dp[j][i] for all j < i
        cur_dp = [0] * i
        cur_argmax = [-1] * i

        for j in range(i):
            gap = primes[i] - primes[j]

            if sorted_struct[j] is not None:
                sorted_gaps, prefix_maxes, prefix_argmax = sorted_struct[j]
                idx = bisect.bisect_right(sorted_gaps, gap) - 1
                if idx >= 0:
                    val = prefix_maxes[idx] + 1
                    cur_dp[j] = val
                    cur_argmax[j] = prefix_argmax[idx]
                else:
                    cur_dp[j] = 2
                    cur_argmax[j] = -1
            else:
                cur_dp[j] = 2
                cur_argmax[j] = -1

            if cur_dp[j] > best_length:
                best_length = cur_dp[j]
                best_i = i
                best_j = j

        # Store results
        dp_table[i] = cur_dp
        argmax_table[i] = cur_argmax

        # Build structure for i: for all k < i, (primes[i]-primes[k], dp[k][i], k)
        if i > 0:
            # Build list of (gap, dp_val, k) for k in range(i)
            pairs = []
            for k in range(i):
                g = primes[i] - primes[k]
                pairs.append((g, cur_dp[k], k))

            pairs.sort(key=lambda x: x[0])
            m = len(pairs)
            sorted_gaps = [p[0] for p in pairs]
            prefix_maxes = [0] * m
            prefix_argmax = [0] * m
            prefix_maxes[0] = pairs[0][1]
            prefix_argmax[0] = pairs[0][2]
            for idx in range(1, m):
                if pairs[idx][1] > prefix_maxes[idx - 1]:
                    prefix_maxes[idx] = pairs[idx][1]
                    prefix_argmax[idx] = pairs[idx][2]
                else:
                    prefix_maxes[idx] = prefix_maxes[idx - 1]
                    prefix_argmax[idx] = prefix_argmax[idx - 1]

            sorted_struct[i] = (sorted_gaps, prefix_maxes, prefix_argmax)

        # Check time
        if time.time() > deadline:
            break

    # Reconstruct the best subsequence
    if best_j < 0:
        return primes[:1]

    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j

    while cur_j >= 0 and argmax_table[cur_i][cur_j] >= 0:
        k = argmax_table[cur_i][cur_j]
        seq.append(primes[k])
        cur_i = cur_j
        cur_j = k

    seq.reverse()

    return seq
