"""Combined program: prime-power baseline with top-K exact matching for merges."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

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

    # Identify potential merges with positive gain
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
                merges.append((gain, p, q, prod))

    # Sort by gain descending and take top K for exact matching
    merges.sort(reverse=True)
    K = 30
    top_merges = merges[:K]
    
    # Build graph from top merges
    merge_primes = set()
    for gain, p, q, prod in top_merges:
        merge_primes.add(p)
        merge_primes.add(q)
    
    prime_list = sorted(merge_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    n_nodes = len(prime_list)
    
    # Build weight matrix
    weight = [[0] * n_nodes for _ in range(n_nodes)]
    merge_product = [[0] * n_nodes for _ in range(n_nodes)]
    
    for gain, p, q, prod in top_merges:
        i = prime_to_idx[p]
        j = prime_to_idx[q]
        weight[i][j] = gain
        weight[j][i] = gain
        merge_product[i][j] = prod
        merge_product[j][i] = prod
    
    # Use DP over subsets if n_nodes is small enough
    if n_nodes <= 16:
        size = 1 << n_nodes
        dp = [-1] * size
        dp[0] = 0
        
        for mask in range(1, size):
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            dp[mask] = dp[remaining]
            
            for j in range(n_nodes):
                if j != i and (remaining & (1 << j)) and weight[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    candidate = dp[new_mask] + weight[i][j]
                    if candidate > dp[mask]:
                        dp[mask] = candidate
        
        full_mask = (1 << n_nodes) - 1
        
        # Reconstruct matching
        matching = []
        mask = full_mask
        while mask:
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            matched_j = -1
            for j in range(n_nodes):
                if j != i and (remaining & (1 << j)) and weight[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    if dp[new_mask] + weight[i][j] == dp[mask]:
                        matched_j = j
                        break
            
            if matched_j != -1:
                matching.append((prime_list[i], prime_list[matched_j], merge_product[i][matched_j]))
                mask = remaining ^ (1 << matched_j)
            else:
                mask = remaining
    else:
        # Fallback to greedy on top merges
        used = set()
        matching = []
        for gain, p, q, prod in top_merges:
            if p not in used and q not in used:
                matching.append((p, q, prod))
                used.add(p)
                used.add(q)
    
    used_primes = set()
    for p, q, prod in matching:
        used_primes.add(p)
        used_primes.add(q)
    
    result = [1]
    for p in primes:
        if p not in used_primes:
            result.append(prime_powers[p])
    
    for p, q, prod in matching:
        result.append(prod)
    
    # Verify
    if len(set(result)) != len(result):
        result = [1] + [prime_powers[p] for p in primes]
    
    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            result = [1] + [prime_powers[p] for p in primes]
            break
    
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

# EVOLVE-BLOCK-END
