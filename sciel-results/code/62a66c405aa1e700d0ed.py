import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def popcount(x):
        return bin(x).count('1')
    
    def has_triangle(adj, n):
        """
        Check if graph has a triangle.
        adj is list of ints (bitmasks).
        """
        for u in range(n):
            # For each edge (u, v) with v > u
            # We can iterate v where bit is set
            v_mask = adj[u] & ((1 << (u + 1)) - 1) # v < u? No, let's just check all
            # Actually, simpler:
            # Triangle exists if there is u < v < w such that u~v, v~w, w~u.
            # For each pair (u, v) with u < v and edge (u, v):
            # Check if adj[u] & adj[v] has any bit set for w > v? Or just any w != u, v.
            # If adj[u] & adj[v] is not 0, there is a common neighbor.
            # If common neighbor w is distinct from u, v, it's a triangle.
            # Since graph is simple, adj[u] does not have bit u.
            # So if adj[u] & adj[v] != 0, there is a w connected to both u and v.
            # This forms a triangle u-v-w.
            # We just need to check if any pair (u, v) has a common neighbor.
            
            # Optimization: Only check pairs (u, v) where u < v.
            # To get v > u, we can mask out bits <= u.
            mask_greater = ~((1 << (u + 1)) - 1)
            candidates = adj[u] & mask_greater
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                # Check common neighbors
                common = adj[u] & adj[v]
                if common:
                    return True
                candidates ^= lsb
        return False

    def max_independent_set_size(adj, n, limit=10):
        """
        Returns True if there is an independent set of size >= limit.
        Uses branch and bound with bitsets.
        """
        # Precompute non-neighbors mask for each vertex
        # non_adj[i] has bit j set if i is NOT connected to j (and i != j)
        non_adj = [0] * n
        for i in range(n):
            # All bits set except i
            all_bits = (1 << n) - 1
            # Remove i
            all_bits ^= (1 << i)
            # Remove neighbors
            non_adj[i] = all_bits & ~adj[i]
        
        found = [False]
        
        def dfs(candidates_mask, current_size):
            if found[0]:
                return
            if current_size >= limit:
                found[0] = True
                return
            
            # Pruning: if not enough candidates left
            if popcount(candidates_mask) < limit - current_size:
                return
            
            # Pick a vertex. Heuristic: pick vertex with max degree in induced subgraph?
            # Or just first one.
            # To speed up, we can pick a vertex that is connected to many others in candidates?
            # For independent set, we want to exclude vertices that block many options.
            # Actually, for finding a LARGE independent set, we should include vertices that are NOT connected to many others?
            # Standard heuristic: pick vertex with max degree in the *complement*? No.
            # Let's just pick the first available vertex.
            
            if candidates_mask == 0:
                return
            
            lsb = candidates_mask & -candidates_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates must be non-neighbors of v, excluding v itself
            new_candidates = candidates_mask & non_adj[v] & ~lsb
            dfs(new_candidates, current_size + 1)
            if found[0]:
                return
            
            # Branch 2: Exclude v
            dfs(candidates_mask & ~lsb, current_size)
            
        dfs((1 << n) - 1, 0)
        return found[0]

    def random_graph(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(n, max_iter=500):
        """
        Try to find a triangle-free graph with alpha < 10 on n vertices.
        Returns edges list if found, else None.
        """
        adj = random_graph(n)
        
        # Step 1: Remove triangles
        # We can do this iteratively
        for _ in range(1000):
            if time.time() > deadline:
                return None
            # Find a triangle
            # We need to find specific triangle to remove an edge
            # Modify has_triangle to return the triangle
            tri = find_triangle(adj, n)
            if not tri:
                break
            u, v, w = tri
            # Remove edge (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Now graph is triangle-free.
        # Check alpha
        if not max_independent_set_size(adj, n, 10):
            # Valid!
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return edges
        
        # If alpha >= 10, try to fix by adding edges
        # This is hard. Just restart?
        # Or try to add an edge between two vertices in the max independent set?
        # We need to find the independent set.
        # Let's just restart with a new random graph.
        
        return None

    def find_triangle(adj, n):
        for u in range(n):
            mask_greater = ~((1 << (u + 1)) - 1)
            candidates = adj[u] & mask_greater
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                common = adj[u] & adj[v]
                if common:
                    # Find w
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    return (u, v, w)
                candidates ^= lsb
        return None

    best_n = 0
    best_edges = []
    
    # Try n=39, 38, ...
    for n in [39, 38, 37, 36, 35, 30, 26]:
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 100 if n > 35 else 500
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = local_search(n)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: n=26 seed graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
