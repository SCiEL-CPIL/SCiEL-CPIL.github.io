import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
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
        
    # Build graph of merges
    # Nodes are indices of primes
    # Edges are (i, j, weight) where i < j
    # Weight = p_i * p_j - P_i - P_j
    # Only if p_i * p_j <= N and weight > 0
    
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    edges = []
    adj = [[] for _ in range(num_primes)]
    
    for i in range(num_primes):
        p = primes[i]
        Pi = prime_powers[p]
        for j in range(i + 1, num_primes):
            q = primes[j]
            Pj = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pi - Pj
            if gain > 0:
                edges.append((gain, i, j))
                adj[i].append((j, gain))
                adj[j].append((i, gain))
                
    # Find Maximum Weight Matching using backtracking
    # We can decompose into connected components to make it faster
    visited = [False] * num_primes
    components = []
    
    for i in range(num_primes):
        if not visited[i]:
            # BFS/DFS to find component
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor, _ in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
            
    # For each component, solve the matching
    # If component size is small, use DP on bitmask?
    # Max component size is likely < 40. 2^40 is too big.
    # Use recursive backtracking with pruning.
    
    def solve_matching(nodes):
        # nodes: list of node indices in this component
        # Map local indices 0..k-1 to global indices
        local_to_global = {local: global_idx for local, global_idx in enumerate(nodes)}
        global_to_local = {global_idx: local for local, global_idx in enumerate(nodes)}
        
        # Build local adjacency list with weights
        local_adj = [[] for _ in range(len(nodes))]
        for i in range(len(nodes)):
            gi = nodes[i]
            for neighbor, gain in adj[gi]:
                if neighbor in global_to_local:
                    j = global_to_local[neighbor]
                    local_adj[i].append((j, gain))
                    
        # Sort nodes by degree descending to improve pruning?
        # Or just use a simple recursive search.
        # State: set of available nodes (bitmask if small, else set)
        # Since size is small, we can use a set or bitmask.
        # Let's use bitmask if len(nodes) <= 25, else set.
        # Actually, let's just use a recursive function that takes a frozenset of available nodes.
        # This might be slow if the set is large.
        
        # Better: Branch and Bound on edges.
        # Sort edges by weight descending.
        local_edges = []
        for i in range(len(nodes)):
            for j, gain in local_adj[i]:
                if i < j:
                    local_edges.append((gain, i, j))
        local_edges.sort(reverse=True)
        
        best_weight = 0
        best_matching = []
        
        def backtrack(available, current_matching, current_weight, edge_index):
            nonlocal best_weight, best_matching
            if current_weight > best_weight:
                best_weight = current_weight
                best_matching = current_matching[:]
                
            # Pruning: even if we take all remaining edges, can we beat best?
            # Upper bound: sum of weights of all remaining edges (loose)
            # Better upper bound: sum of weights of edges incident to available nodes?
            
            if edge_index >= len(local_edges):
                return
                
            gain, i, j = local_edges[edge_index]
            
            # Option 1: Skip this edge
            backtrack(available, current_matching, current_weight, edge_index + 1)
            
            # Option 2: Take this edge if i and j are available
            if i in available and j in available:
                new_available = available - {i, j}
                current_matching.append((i, j, gain))
                backtrack(new_available, current_matching, current_weight + gain, edge_index + 1)
                current_matching.pop()
                
        initial_available = set(range(len(nodes)))
        backtrack(initial_available, [], 0, 0)
        
        # Convert local matching to global
        global_matching = []
        for i, j, gain in best_matching:
            gi = local_to_global[i]
            gj = local_to_global[j]
            global_matching.append((gi, gj, gain))
            
        return global_matching

    all_matching = []
    for comp in components:
        if len(comp) < 2:
            continue
        if time.time() > deadline - 5:
            # Fallback to greedy if running out of time
            break
        matching = solve_matching(comp)
        all_matching.extend(matching)
        
    # If timeout during exact solve, fallback to greedy
    if time.time() > deadline - 5:
        # Simple greedy fallback
        used_primes = set()
        selected_merges = []
        # Re-sort edges by gain
        edges.sort(reverse=True)
        for gain, i, j in edges:
            if i not in used_primes and j not in used_primes:
                selected_merges.append((i, j, gain))
                used_primes.add(i)
                used_primes.add(j)
        all_matching = selected_merges
    else:
        # Convert all_matching to list of (i, j, gain)
        # It is already in that format from solve_matching
        pass

    # Construct final result
    used_primes = set()
    for i, j, gain in all_matching:
        used_primes.add(i)
        used_primes.add(j)
        
    result = [1]
    for i, p in enumerate(primes):
        if i not in used_primes:
            result.append(prime_powers[p])
            
    # Add the merged products
    for i, j, gain in all_matching:
        p = primes[i]
        q = primes[j]
        result.append(p * q)
        
    # Verify
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
