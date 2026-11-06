import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Leave 10s margin for 120s limit

    def time_check():
        return time.time() - start_time

    def check_triangle_free(n, edges):
        """
        Returns True if the graph is triangle-free.
        Uses bitsets for speed.
        """
        # Build adjacency masks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        # Check triangle-free
        # If (u,v) is an edge, adj[u] & adj[v] must be 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    if adj_masks[u] & adj_masks[v]:
                        return False
        return True

    def max_independent_set_size(n, edges):
        """
        Returns the size of the maximum independent set.
        Uses bitsets and branch and bound.
        """
        # Build adjacency masks
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        best = [0]
        
        def dfs(candidates, size):
            if time_check() > DEADLINE - 5:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            # Upper bound
            # Count bits in candidates
            # In Python 3.10+ int.bit_count() is available
            # Assuming modern Python
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Pick a vertex
            # Strategy: pick vertex with highest degree in the subgraph defined by candidates?
            # To simplify, pick the lowest set bit.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def is_valid(n, edges):
        """
        Returns True if the graph is triangle-free and has no independent set of 10 vertices.
        """
        if not check_triangle_free(n, edges):
            return False
        alpha = max_independent_set_size(n, edges)
        return alpha < 10

    def search_graph(n, seed):
        random.seed(seed)
        # Initialize edges
        edges = []
        adj_masks = [0] * n
        
        # Try to add edges randomly
        # Order of vertices to consider for edges
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        # Phase 1: Random initialization
        for u, v in pairs:
            if adj_masks[u] & (1 << v): continue
            # Check if adding (u,v) creates triangle
            if adj_masks[u] & adj_masks[v]:
                continue
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
        # Check current alpha
        alpha = max_independent_set_size(n, edges)
        if alpha < 10:
            return n, edges

        # Phase 2: Greedy improvement
        # Try to add more edges to reduce alpha
        # We will iterate through potential edges and add them if they don't create triangles
        # and if they don't increase alpha (or reduce it)
        # Actually, adding edges never increases alpha.
        # So we can just try to add all possible edges that don't create triangles.
        # But we might want to be selective if we get stuck?
        # No, if adding an edge doesn't help reduce alpha below 10, maybe a different combination does?
        # But since we just want ANY valid graph, adding all safe edges is a good move to minimize alpha.
        
        # Let's try adding all safe edges
        for u, v in pairs:
            if time_check() > DEADLINE - 10:
                break
            if adj_masks[u] & (1 << v): continue
            if adj_masks[u] & adj_masks[v]:
                continue
            # Potential edge (u,v)
            # Add it temporarily
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
            # Check alpha
            new_alpha = max_independent_set_size(n, edges)
            if new_alpha < 10:
                return n, edges
                
            # If alpha didn't drop below 10, keep the edge anyway?
            # Yes, because it might help with subsequent edge additions?
            # Actually, a denser graph has smaller alpha.
            # So keeping it is generally good.
            
        # If we added all safe edges and still alpha >= 10, this seed failed
        return None, None

    # Try to find a graph for n=39, then 38, etc.
    best_n = 0
    best_edges = []
    
    # We can try a few seeds for each n
    # Since n=39 is the target, spend most time there.
    
    # First, check if we can get n=39
    for seed in range(1000):
        if time_check() > DEADLINE - 10:
            break
        n_found, edges_found = search_graph(39, seed)
        if n_found:
            best_n = 39
            best_edges = edges_found
            break
        if time_check() > DEADLINE - 30:
            break
            
    if best_n == 0:
        # Try n=38
        for seed in range(1000):
            if time_check() > DEADLINE - 10:
                break
            n_found, edges_found = search_graph(38, seed)
            if n_found:
                best_n = 38
                best_edges = edges_found
                break
            if time_check() > DEADLINE - 20:
                break
                
    if best_n == 0:
        # Fallback to known small graph (Andrasfai And(9) on 26 vertices)
        # Just in case the search fails completely
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
        best_edges = edges

    return {'n': best_n, 'edges': best_edges}
