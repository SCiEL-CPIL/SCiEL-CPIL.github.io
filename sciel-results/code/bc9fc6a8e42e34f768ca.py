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
        
    # Identify valid edges for merging
    # A merge of p and q is beneficial if p*q > M_p + M_q
    n = len(primes)
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
                valid_edges.append((i, j, g))
                
    # If no beneficial merges, return all max prime powers plus 1
    if not valid_edges:
        result = [1] + [prime_powers[p] for p in primes]
        return result

    # Extract subgraph of nodes involved in beneficial merges
    nodes_in_edges = set()
    for i, j, g in valid_edges:
        nodes_in_edges.add(i)
        nodes_in_edges.add(j)
        
    subgraph_nodes = sorted(list(nodes_in_edges))
    node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
    m = len(subgraph_nodes)
    
    # Build weight matrix for the subgraph
    # w[i][j] = gain if edge exists, else -1
    w = [[-1]*m for _ in range(m)]
    for i, j, g in valid_edges:
        ui = node_to_idx[i]
        uj = node_to_idx[j]
        w[ui][uj] = g
        w[uj][ui] = g
        
    from functools import lru_cache
    
    # DP for maximum weight matching
    # mask: bitmask of available nodes in the subgraph
    # Returns (max_gain, tuple_of_pairs)
    # Pairs are (subgraph_idx_i, subgraph_idx_j)
    
    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0, ()
        
        # Find first available node
        i = 0
        while i < m:
            if mask & (1 << i):
                break
            i += 1
            
        if i >= m:
            return 0, ()
            
        rem_mask = mask ^ (1 << i)
        
        # Option 1: Leave i unmatched
        val1, pairs1 = dp(rem_mask)
        best_val = val1
        best_pairs = pairs1
        
        # Option 2: Match i with some j in rem_mask
        # We only consider j > i to avoid double counting symmetric pairs?
        # Actually, since we pick the smallest i, any valid matching will have i as the smallest
        # in its pair if we enforce j > i. 
        # However, the standard bitmask matching DP usually just iterates all j in mask.
        # To be safe and consistent with the "pick smallest i" strategy:
        # If we match i with j, j must be in rem_mask.
        # Since we always pick the smallest available i, we don't need to restrict j > i 
        # explicitly for correctness of the set of matchings, but it reduces states?
        # No, the state is defined by the mask.
        # If we match i with j, the new mask is rem_mask ^ (1 << j).
        # This covers all possibilities.
        
        j = 0
        while j < m:
            if rem_mask & (1 << j):
                if w[i][j] > 0: # Edge exists and is beneficial
                    val2, pairs2 = dp(rem_mask ^ (1 << j))
                    total = w[i][j] + val2
                    if total > best_val:
                        best_val = total
                        best_pairs = pairs2 + ((i, j),)
            j += 1
            
        return best_val, best_pairs
        
    if not time_check():
        # Fallback if DP is too slow (unlikely for m < 20)
        return [1] + [prime_powers[p] for p in primes]

    full_mask = (1 << m) - 1
    _, selected_pairs_sub = dp(full_mask)
    
    # Map subgraph indices back to original prime indices
    selected_prime_pairs = []
    for i_sub, j_sub in selected_pairs_sub:
        i_orig = subgraph_nodes[i_sub]
        j_orig = subgraph_nodes[j_sub]
        selected_prime_pairs.append((i_orig, j_orig))
        
    # Construct result
    result = [1]
    
    # Set of original prime indices that are matched
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
        # Fallback
        return [1] + [prime_powers[p] for p in primes]
        
    for x in result:
        if x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]
            
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]
                
    return result
