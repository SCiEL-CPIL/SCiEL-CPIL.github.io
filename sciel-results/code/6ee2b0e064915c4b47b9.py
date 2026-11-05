import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def is_triangle_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if common:
                        return False
        return True

    def max_independent_set_size(adj, n):
        # Branch and bound to find max independent set size
        # Use bitmasks for speed
        best = [0]
        
        def dfs(candidates, current_size, current_set):
            if current_size > best[0]:
                best[0] = current_size
            if not candidates:
                return
            # Pruning: if current_size + popcount(candidates) <= best[0], return
            if current_size + bin(candidates).count('1') <= best[0]:
                return
            
            # Pick a vertex with minimum degree in the induced subgraph?
            # Or just the first one.
            # Let's pick the vertex with the smallest index for simplicity first.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates: candidates that are not neighbors of v
            new_candidates = candidates & ~(adj[v] | (1 << v))
            dfs(new_candidates, current_size + 1, current_set | (1 << v))
            
            # Branch 2: Exclude v
            dfs(candidates & ~lsb, current_size, current_set)

        dfs((1 << n) - 1, 0, 0)
        return best[0]

    def find_independent_set_of_size_k(adj, n, k):
        found = [None]
        
        def dfs(candidates, current_set):
            if found[0] is not None:
                return
            if len(current_set) == k:
                found[0] = current_set
                return
            if bin(candidates).count('1') < k - len(current_set):
                return
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            new_candidates = candidates & ~(adj[v] | (1 << v))
            dfs(new_candidates, current_set | (1 << v))
            
            # Exclude v
            dfs(candidates & ~lsb, current_set)

        dfs((1 << n) - 1, 0)
        return found[0]

    def try_improve_alpha(adj, n, target_alpha=10):
        # Try to add edges to reduce alpha
        # We want alpha < target_alpha
        # If alpha >= target_alpha, find an independent set of size target_alpha
        # and try to add an edge within it that doesn't create a triangle.
        
        iso = find_independent_set_of_size_k(adj, n, target_alpha)
        if iso is None:
            return True, adj # Already valid
        
        # Try to add an edge between two vertices in iso
        # Check all pairs
        improved = False
        for i in range(target_alpha):
            for j in range(i + 1, target_alpha):
                u = (iso >> i) & 1
                v = (iso >> j) & 1
                if u and v:
                    # Check if edge (i, j) can be added
                    if not (adj[i] & (1 << j)):
                        common = adj[i] & adj[j]
                        if common == 0:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
                            break
            if improved:
                break
        return improved, adj

    # Strategy:
    # 1. Generate a candidate graph using a known algebraic construction.
    # The "shift" graph: Vertices are Z_n. Edge (i, j) if j-i mod n in S.
    # For triangle-free, S must not contain x, y, x+y.
    # A common choice for R(3, k) constructions involves sum-free sets.
    
    # Let's try a specific known structure: The complement of the line graph of K_5?
    # No, let's stick to the cyclic group construction which is robust.
    
    # We want n=39.
    # Let's try a few different shift sets S for n=39.
    # S must be symmetric: if x in S, then -x in S.
    # S must be sum-free: no x, y in S such that x+y in S.
    
    # For n=39, let's try to find a large sum-free set S.
    # Max size of sum-free set in Z_n is roughly n/2.
    # If |S| is large, the graph is dense.
    # But we need triangle-free.
    
    # Let's use a heuristic search over possible shift sets.
    # Or better, use a known valid graph if we can construct it.
    
    # The seed program uses a fallback to n=26.
    # Let's try to construct n=39 using a random restart local search but with a better initialization.
    # Initialization: Start with a bipartite graph? No, bipartite graphs have alpha >= n/2.
    # For n=39, alpha >= 20. We need alpha < 10.
    # So we need a very dense triangle-free graph.
    
    # The extremal triangle-free graphs are close to Turan graphs T_3(n).
    # T_3(39) has 3 parts of size 13.
    # T_3(39) is triangle-free.
    # What is its alpha?
    # In T_3(n), the maximum independent set is the largest part, size ceil(n/3).
    # For n=39, alpha = 13.
    # We need alpha < 10.
    # So we need to modify T_3(39) to reduce alpha.
    # Adding edges within the parts? No, that creates triangles.
    # Removing edges between parts? That increases alpha.
    # Wait, T_3(n) is complete 3-partite.
    # Any independent set is contained in one part.
    # So alpha(T_3(39)) = 13.
    # To reduce alpha, we must "break" the parts.
    # This suggests the graph is not a Turan graph.
    
    # The known 39-vertex graph is likely a "blow-up" of a smaller triangle-free graph with small alpha.
    # Clebsch graph: n=16, alpha=5.
    # Blow up by k=2? n=32, alpha=10. (Too large, need <10).
    # Blow up by k=1? n=16.
    
    # What about the Hoffman-Singleton graph? n=50, alpha=7.
    # Subgraph of size 39?
    
    # Let's try a random local search starting from a random graph, but with a strong bias towards
    # maintaining triangle-freeness and minimizing alpha.
    
    n = 39
    full_mask = (1 << n) - 1
    best_result = None
    best_n = 0
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with a random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Make triangle-free
        # Repeatedly remove edges from triangles
        for _ in range(1000):
            if time.time() > deadline:
                break
            tri_found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        common = adj[u] & adj[v]
                        if common:
                            lsb = common & -common
                            w = lsb.bit_length() - 1
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            tri_found = True
                            break
                if tri_found:
                    break
            if not tri_found:
                break
        
        # Phase 2: Reduce Alpha
        # We want alpha < 10
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Check if alpha < 10
            # If yes, we have a valid graph
            iso = find_independent_set_of_size_k(adj, n, 10)
            if iso is None:
                # Valid
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'edges': edges}
                break
            else:
                # Try to add an edge to break this independent set
                improved, adj = try_improve_alpha(adj, n, 10)
                if not improved:
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

import random
