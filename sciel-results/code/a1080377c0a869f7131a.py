import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    
    # 1. Sieve primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate M_p
    M = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        M[p] = power
        
    # 3. Find all positive gain pairs
    # Edge: (p, q, gain, product)
    edges = []
    for i in range(len(primes)):
        p = primes[i]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            product = p * q
            if product > N:
                break # Since primes are sorted, further q will be larger
            gain = product - (M[p] + M[q])
            if gain > 0:
                edges.append((p, q, gain, product))
                
    # 4. Maximum Weight Matching
    # The graph is small. We can use a simple recursive search or DP over subsets if small enough.
    # Number of primes is 54. But only primes involved in edges matter for the matching.
    # Let's identify the set of primes involved in edges.
    involved_primes = set()
    for p, q, g, prod in edges:
        involved_primes.add(p)
        involved_primes.add(q)
        
    # If the number of involved primes is small, we can do a bitmask DP or simple backtracking.
    # Let's map involved primes to indices 0..K-1
    prime_to_idx = {p: i for i, p in enumerate(sorted(involved_primes))}
    K = len(involved_primes)
    
    # If K is large, we might need a better algorithm, but K is likely small (< 20).
    # Let's check K.
    # Primes with positive gain pairs:
    # 5, 7, 11, 13 and their partners.
    # Partners of 5: 17, 19, 23, 29, 31, 37, 41, 43, 47
    # Partners of 7: 17, 19, 23, 29, 31
    # Partners of 11: 17, 19, 23
    # Partners of 13: 17, 19
    # Union: {5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    # Size is about 13. 2^13 = 8192 states. Very fast.
    
    if K <= 20:
        # DP over subsets of involved primes
        # dp[mask] = max gain achievable using subset of primes indicated by mask
        # We iterate through edges and update dp.
        # Actually, standard MW matching on general graph is hard, but this is a bipartite-like structure? No.
        # But K is small. We can just do backtracking over edges sorted by weight.
        
        # Sort edges by gain descending
        edges.sort(key=lambda x: x[2], reverse=True)
        
        best_gain = 0
        best_mask = 0 # Bitmask of used primes
        
        def backtrack(edge_idx, used_mask, current_gain):
            nonlocal best_gain, best_mask
            if time.time() > deadline:
                return
            if current_gain > best_gain:
                best_gain = current_gain
                best_mask = used_mask
                
            if edge_idx == len(edges):
                return
                
            # Option 1: Skip edge
            backtrack(edge_idx + 1, used_mask, current_gain)
            
            # Option 2: Take edge
            p, q, gain, product = edges[edge_idx]
            idx_p = prime_to_idx[p]
            idx_q = prime_to_idx[q]
            bit_p = 1 << idx_p
            bit_q = 1 << idx_q
            
            if not (used_mask & bit_p) and not (used_mask & bit_q):
                backtrack(edge_idx + 1, used_mask | bit_p | bit_q, current_gain + gain)
                
        backtrack(0, 0, 0)
        
        # Reconstruct selected pairs from best_mask?
        # Actually, we just need to know which pairs were selected.
        # It's easier to store the selected edges in the backtrack.
        
        best_selected_edges = []
        
        def backtrack2(edge_idx, used_mask, current_gain, selected):
            nonlocal best_gain, best_selected_edges
            if time.time() > deadline:
                return
            if current_gain > best_gain:
                best_gain = current_gain
                best_selected_edges = list(selected)
                
            if edge_idx == len(edges):
                return
                
            # Option 1: Skip edge
            backtrack2(edge_idx + 1, used_mask, current_gain, selected)
            
            # Option 2: Take edge
            p, q, gain, product = edges[edge_idx]
            idx_p = prime_to_idx[p]
            idx_q = prime_to_idx[q]
            bit_p = 1 << idx_p
            bit_q = 1 << idx_q
            
            if not (used_mask & bit_p) and not (used_mask & bit_q):
                selected.append((p, q, product))
                backtrack2(edge_idx + 1, used_mask | bit_p | bit_q, current_gain + gain, selected)
                selected.pop()
                
        backtrack2(0, 0, 0, [])
    else:
        # Fallback to greedy if K is too large (unlikely)
        used = set()
        best_selected_edges = []
        for p, q, gain, product in edges:
            if p not in used and q not in used:
                best_selected_edges.append((p, q, product))
                used.add(p)
                used.add(q)

    # 5. Construct Result
    result = [1]
    used_primes = set()
    for p, q, product in best_selected_edges:
        result.append(product)
        used_primes.add(p)
        used_primes.add(q)
        
    for p in primes:
        if p not in used_primes:
            result.append(M[p])
            
    # 6. Verification
    if len(set(result)) != len(result):
        result = [1] + [M[p] for p in primes]
        
    valid = True
    for x in result:
        if x < 1 or x > N:
            valid = False
            break
    if not valid:
        result = [1] + [M[p] for p in primes]
        
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        result = [1] + [M[p] for p in primes]

    return [int(x) for x in result]
