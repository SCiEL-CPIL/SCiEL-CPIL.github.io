import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute for the backtracking approach
    # For n vertices, we build vertex by vertex.
    # When adding vertex k (0-indexed), we choose its adjacency to vertices 0..k-1.
    # The adjacency must not create a K4 or an I6.
    
    # Key insight: A K4 is formed when vertex k connects to 3 vertices that form a triangle.
    # An I6 is formed when vertex k is non-adjacent to 5 vertices that form an I5.
    
    # We can precompute for each partial graph the "forbidden patterns" for the next vertex.
    # A pattern (subset of {0,...,k-1}) is forbidden if:
    #   1. It contains 3 vertices that form a triangle in the current graph (would create K4)
    #   2. Its complement in {0,...,k-1} contains 5 vertices that form an I5 (would create I6)
    
    # For efficiency, we'll use a randomized backtracking with memoization of dead-ends.
    
    def popcount(x):
        return bin(x).count('1')
    
    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def has_k4_in_subgraph(adj, vertices, n):
        """Check if the subgraph induced by 'vertices' contains a K4."""
        k = len(vertices)
        if k < 4:
            return False
        # For small k, brute force
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j+1, k):
                    for m in range(l+1, k):
                        vi, vj, vl, vm = vertices[i], vertices[j], vertices[l], vertices[m]
                        if ((adj[vi] >> vj) & 1) and ((adj[vi] >> vl) & 1) and ((adj[vi] >> vm) & 1) and \
                           ((adj[vj] >> vl) & 1) and ((adj[vj] >> vm) & 1) and \
                           ((adj[vl] >> vm) & 1):
                            return True
        return False
    
    def has_i6_in_subgraph(adj, vertices, n):
        """Check if the subgraph induced by 'vertices' contains an independent set of 6."""
        k = len(vertices)
        if k < 6:
            return False
        # Check if there's an independent set of 6 in the complement
        # For small k, we can use a simple check
        from itertools import combinations
        for combo in combinations(vertices, 6):
            is_independent = True
            for a in range(6):
                for b in range(a+1, 6):
                    if (adj[combo[a]] >> combo[b]) & 1:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return True
        return False
    
    # Strategy: Try to build a valid graph with n=25, 30, 35 using backtracking
    # with randomized vertex ordering and adjacency choices.
    
    # For better performance, we'll use a different approach:
    # Start with a known good structure and try to extend it.
    
    # Actually, let's try a simpler but more effective approach:
    # Use simulated annealing with a better cost function and more aggressive search.
    
    def count_k4_fast(adj, n):
        """Count K4s using bit operations."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                common = adj[i] & adj[j]
                # Count edges in the common neighborhood
                # For each pair (k,l) in common with k < l, check if edge exists
                k_mask = common
                while k_mask:
                    lb = k_mask & (-k_mask)
                    k_idx = lb.bit_length() - 1
                    k_mask ^= lb
                    if k_idx <= j:
                        continue
                    l_mask = common & ((1 << n) - 1)
                    l_mask &= ~((1 << (k_idx + 1)) - 1)  # Only l > k_idx
                    count += popcount(adj[k_idx] & l_mask)
        return count
    
    def count_i6_fast(adj, n):
        """Count I6s by counting K6s in the complement graph."""
        mask = (1 << n) - 1
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    continue  # Edge in G, so not connected in complement
                common_non = (~adj[i]) & (~adj[j]) & mask
                common_non &= ~((1 << (j + 1)) - 1)  # Only vertices > j
                for k in range(j+1, n):
                    if k >= n:
                        break
                    if (adj[i] >> k) & 1 or (adj[j] >> k) & 1:
                        continue
                    common_non2 = common_non & ((1 << n) - 1)
                    common_non2 &= ~((1 << (k + 1)) - 1)  # Only vertices > k
                    for l in range(k+1, n):
                        if l >= n:
                            break
                        if (adj[i] >> l) & 1 or (adj[j] >> l) & 1 or (adj[k] >> l) & 1:
                            continue
                        common_non3 = common_non2 & ((1 << n) - 1)
                        common_non3 &= ~((1 << (l + 1)) - 1)  # Only vertices > l
                        count += popcount(common_non3)
        return count
    
    best_n = 16
    best_edges = None
    
    # First, ensure we have a valid base graph
    # Use a random graph with appropriate density
    for target_n in [25, 20, 17]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            p = 0.35
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_cost = count_k4_fast(adj, target_n) + count_i6_fast(adj, target_n)
            
            # Simulated annealing
            T = 2.0
            cooling = 0.9995
            min_T = 0.05
            steps = 0
            max_steps = 50000
            
            while T > min_T and steps < max_steps and time.time() < deadline - 5:
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                was_edge = (adj[u] >> v) & 1
                if was_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = count_k4_fast(adj, target_n) + count_i6_fast(adj, target_n)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost == 0:
                        found = True
                        break
                else:
                    if was_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                
                T *= cooling
                steps += 1
            
            if found:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = to_edges(adj, target_n)
                break
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a small valid graph
    # Paley graph of order 17 is K4-free and has no I6
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': to_edges(adj, 17)}
