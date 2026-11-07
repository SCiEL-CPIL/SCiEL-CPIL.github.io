import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, find the largest prime power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Baseline sum
    baseline_sum = 1 + sum(prime_powers[p] for p in primes)

    # Identify beneficial merges
    # (gain, p, q)
    merges = []
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                merges.append((gain, p, q))

    # Identify active primes (those involved in any merge)
    active_primes_set = set()
    for gain, p, q in merges:
        active_primes_set.add(p)
        active_primes_set.add(q)
    
    active_primes = sorted(active_primes_set)
    prime_to_idx = {p: i for i, p in enumerate(active_primes)}
    k = len(active_primes)
    
    # Build weight matrix for active primes
    # weight[i][j] = gain if merge exists, else -infinity (or 0 if we handle logic carefully)
    # We want max weight matching.
    # DP state: mask of active primes used.
    # dp[mask] = max gain from matching a subset of primes indicated by mask.
    # Actually, let's define dp[mask] as max gain obtainable considering only primes in mask, 
    # where mask is the set of primes *available* to be matched.
    # Or simpler: dp[mask] = max gain from a matching on the subgraph induced by vertices in mask.
    # Transition: pick lowest bit v.
    # 1. v is unmatched: dp[mask without v]
    # 2. v is matched with u in mask: weight[v][u] + dp[mask without v and u]
    
    if k <= 20:
        size = 1 << k
        dp = [0] * size
        
        # Precompute weights for active primes
        weights = [[0] * k for _ in range(k)]
        for gain, p, q in merges:
            i = prime_to_idx[p]
            j = prime_to_idx[q]
            weights[i][j] = gain
            weights[j][i] = gain
            
        for mask in range(1, size):
            # Find lowest set bit
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            # Option 1: i is unmatched
            dp[mask] = dp[remaining]
            
            # Option 2: i is matched with some j in remaining
            for j in range(k):
                if (remaining & (1 << j)) and weights[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    candidate = weights[i][j] + dp[new_mask]
                    if candidate > dp[mask]:
                        dp[mask] = candidate
        
        full_mask = (1 << k) - 1
        max_gain = dp[full_mask]
        
        # Reconstruct matching
        matching = []
        mask = full_mask
        while mask:
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            matched_j = -1
            for j in range(k):
                if (remaining & (1 << j)) and weights[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    if weights[i][j] + dp[new_mask] == dp[mask]:
                        matched_j = j
                        break
            
            if matched_j != -1:
                matching.append((active_primes[i], active_primes[matched_j]))
                mask = remaining ^ (1 << matched_j)
            else:
                mask = remaining
    else:
        # Fallback to greedy if too many active primes (unlikely)
        used = set()
        matching = []
        merges_sorted = sorted(merges, reverse=True)
        for gain, p, q in merges_sorted:
            if p not in used and q not in used:
                matching.append((p, q))
                used.add(p)
                used.add(q)
        # Calculate gain for greedy
        max_gain = 0
        for p, q in matching:
            max_gain += p * q - prime_powers[p] - prime_powers[q]

    # Construct result
    matched_primes = set()
    for p, q in matching:
        matched_primes.add(p)
        matched_primes.add(q)
        
    result = [1]
    for p in primes:
        if p not in matched_primes:
            result.append(prime_powers[p])
            
    for p, q in matching:
        result.append(p * q)
        
    # Verify
    if len(set(result)) != len(result):
        result = [1] + [prime_powers[p] for p in primes]
    else:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result = [1] + [prime_powers[p] for p in primes]
            
    result = [int(x) for x in result]
    return result
