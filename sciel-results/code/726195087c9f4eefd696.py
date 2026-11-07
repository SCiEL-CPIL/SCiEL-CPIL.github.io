import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    def time_check():
        return time.time() < deadline
        
    N = 256
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Identify primes that can be part of a beneficial merge
    # A merge of p and q is beneficial if p*q <= N and p*q > M_p + M_q
    # We build a graph of such merges.
    
    # Map primes to indices for the subgraph
    # Only primes p where M_p < p*q for some q are candidates.
    # Actually, just check all pairs.
    
    n = len(primes)
    # gain[i][j]
    gain = [[0]*n for _ in range(n)]
    valid_edges = []
    
    for i in range(n):
        p = primes[i]
        for j in range(i+1, n):
            q = primes[j]
            prod = p * q
            if prod > N:
                break
            g = prod - prime_powers[p] - prime_powers[q]
            if g > 0:
                gain[i][j] = g
                gain[j][i] = g
                valid_edges.append((i, j, g))
                
    # Find nodes involved in valid edges
    nodes_in_edges = set()
    for i, j, g in valid_edges:
        nodes_in_edges.add(i)
        nodes_in_edges.add(j)
        
    if not nodes_in_edges:
        result = [1] + [prime_powers[p] for p in primes]
        return result

    # Map original indices to compact indices for bitmask DP
    subgraph_nodes = sorted(list(nodes_in_edges))
    node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
    m = len(subgraph_nodes)
    
    # Build weight matrix for the subgraph
    # w[i][j] = gain if edge exists, else 0 (or -1 to indicate no edge)
    # We only care about positive gains.
    w = [[0]*m for _ in range(m)]
    for i, j, g in valid_edges:
        ui = node_to_idx[i]
        uj = node_to_idx[j]
        w[ui][uj] = g
        w[uj][ui] = g
        
    # DP over bitmask of subgraph_nodes
    # dp[mask] = max gain obtainable from the set of nodes indicated by mask
    # mask has 1s for nodes that are STILL AVAILABLE.
    
    full_mask = (1 << m) - 1
    
    if not time_check():
        return [1] + [prime_powers[p] for p in primes]
    
    # Memoized DP for maximum weight matching
    memo = {}
    
    def dp2(mask):
        if mask in memo:
            return memo[mask]
        if mask == 0:
            memo[mask] = (0, ())
            return memo[mask]
        
        i = 0
        while i < m:
            if mask & (1 << i):
                break
            i += 1
            
        if i >= m:
            memo[mask] = (0, ())
            return memo[mask]
            
        rem_mask = mask ^ (1 << i)
        
        # Option 1: Unmatched
        val1, pairs1 = dp2(rem_mask)
        best_val = val1
        best_pairs = pairs1
        
        # Option 2: Matched
        j = i + 1
        while j < m:
            if rem_mask & (1 << j):
                if w[i][j] > 0:
                    val2, pairs2 = dp2(rem_mask ^ (1 << j))
                    total = w[i][j] + val2
                    if total > best_val:
                        best_val = total
                        best_pairs = pairs2 + ((i, j),)
            j += 1
            
        memo[mask] = (best_val, best_pairs)
        return memo[mask]
        
    _, selected_pairs_sub = dp2(full_mask)
    
    # Map subgraph indices back to original prime indices
    selected_prime_pairs = []
    for i_sub, j_sub in selected_pairs_sub:
        i_orig = subgraph_nodes[i_sub]
        j_orig = subgraph_nodes[j_sub]
        selected_prime_pairs.append((i_orig, j_orig))
        
    # Construct result
    result = [1]
    
    matched_indices = set()
    for i, j in selected_prime_pairs:
        matched_indices.add(i)
        matched_indices.add(j)
        
    # Add unmatched primes with their M_p
    for i, p in enumerate(primes):
        if i not in matched_indices:
            result.append(prime_powers[p])
            
    # Add matched products
    for i, j in selected_prime_pairs:
        p = primes[i]
        q = primes[j]
        result.append(p * q)
        
    # Final checks
    if len(set(result)) != len(result):
        return [1] + [prime_powers[p] for p in primes]
        
    for x in result:
        if x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]
            
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]
                
    return result
