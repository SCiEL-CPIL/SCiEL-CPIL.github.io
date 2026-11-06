import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds; leave safety margin before 120s limit

    def check_triangle_free(n, edges):
        """Check if graph is triangle-free using adjacency matrix."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def has_independent_set_of_size_k(n, edges, k):
        """
        Returns True if there is an independent set of size k.
        Uses branch and bound.
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # We want to find if max independent set >= k
        # State: (current_size, candidates)
        # candidates: set of vertices that can be added to the current independent set
        
        def dfs(size, candidates):
            if size == k:
                return True
            if not candidates:
                return False
            # Upper bound check
            if size + len(candidates) < k:
                return False
            
            # Heuristic: pick a vertex from candidates
            # To prune effectively, we can just pick one. 
            # Picking a vertex with high degree in the induced subgraph might help?
            # Or just the first one.
            v = next(iter(candidates))
            
            # Branch 1: Include v
            # New candidates: candidates \ {v} \ N(v)
            new_cands_incl = candidates - {v} - adj[v]
            if dfs(size + 1, new_cands_incl):
                return True
            
            # Branch 2: Exclude v
            # New candidates: candidates \ {v}
            new_cands_excl = candidates - {v}
            if dfs(size, new_cands_excl):
                return True
            
            return False

        return dfs(0, set(range(n)))

    def is_valid(n, edges):
        """Check both conditions: no triangle and no independent set of 10."""
        if time.time() - start_time > DEADLINE - 5:
            return False
        if not check_triangle_free(n, edges):
            return False
        # Check if alpha < 10, i.e., no independent set of size 10
        if has_independent_set_of_size_k(n, edges, 10):
            return False
        return True

    best_n = 0
    best_edges = []

    # Try circulant graphs C_n(1 mod 3) for n from 40 down to 27
    # These are triangle-free if 3 does not divide n (for n >= 10)
    for n in range(40, 26, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Construct edges for C_n(1 mod 3)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    edges.append([i, j])
        
        # Quick triangle check is implicitly handled by the property, 
        # but is_valid checks it anyway.
        # For n not divisible by 3, this graph is triangle-free.
        if n % 3 != 0:
            if is_valid(n, edges):
                best_n = n
                best_edges = edges
                print(f"Found valid circulant graph with n={n}")
                return {'n': best_n, 'edges': best_edges}
        else:
            # If n is divisible by 3, it might have triangles. 
            # Let's check just in case, though unlikely to be valid if triangles exist.
            if is_valid(n, edges):
                best_n = n
                best_edges = edges
                print(f"Found valid graph with n={n}")
                return {'n': best_n, 'edges': best_edges}

    # Fallback to known valid graph: Andrasfai And(9) on 26 vertices
    n1 = 26
    edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
    
    # Verify it's still valid (it should be)
    if is_valid(n1, edges1):
        best_n = n1
        best_edges = edges1
    else:
        # Emergency fallback: empty graph on 1 vertex? 
        # But we need alpha < 10. K_1 has alpha=1.
        # Actually, if And(9) fails, something is wrong.
        # Let's just return a small valid graph.
        # K_2 is triangle free, alpha=1.
        best_n = 2
        best_edges = [[0, 1]]

    return {'n': best_n, 'edges': best_edges}
