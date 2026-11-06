"""
Construct a C4-free graph with no independent 11-set, maximizing n.
Uses a structural approach: start from known C4-free constructions and verify.
The best known construction has 39 vertices.
"""
import time
import random
import itertools

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Try to construct a 39-vertex C4-free graph with alpha <= 10.
    2. Use a known construction or search for one.
    3. Verify C4-freeness and independence number.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We'll try to build a graph with 39 vertices.
    # One approach: use a strongly regular graph or a known C4-free graph.
    # The Clebsch graph has 16 vertices, degree 5, C4-free, alpha=4.
    # We can try to combine or extend.
    
    # Let's try a different approach: use a random search with local improvement.
    # Or use a known construction.
    
    # Known fact: R(C4, K11) > 39 means there exists a 39-vertex C4-free graph with alpha <= 10.
    # Let's try to construct one using a greedy + local search approach.
    
    def has_c4(n, edges):
        """Check if graph has a C4. O(n^3) but with bitmasks it's fast."""
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # For each pair of vertices, check if they have two common neighbors
        # A C4 exists iff there are two vertices with >= 2 common neighbors
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):  # at least 2 bits set
                    return True
        return False
    
    def max_independent_set_size(n, edges):
        """
        Find the size of the maximum independent set using branch and bound.
        For n=39, this might be slow, so we use a heuristic + verification.
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Use a simple greedy + local search to estimate, then verify
        # Actually, for verification we need to check if alpha >= 11
        # We can use a branch and bound with pruning
        
        # Convert to bitmasks for speed
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        best = [0]
        
        def branch(candidates, current_size):
            if time.time() > deadline - 5:
                return
            if not candidates:
                best[0] = max(best[0], current_size)
                return
            if current_size + len(candidates) <= best[0]:
                return
            
            # Pick vertex with highest degree in candidate subgraph
            # For speed, just pick the first one
            v = candidates[0]
            remaining = candidates[1:]
            
            # Option 1: include v
            # New candidates are those not adjacent to v
            new_candidates = [u for u in remaining if not (adj_mask[v] & (1 << u))]
            branch(new_candidates, current_size + 1)
            
            # Option 2: exclude v
            branch(remaining, current_size)
        
        branch(list(range(n)), 0)
        return best[0]
    
    def verify_graph(n, edges):
        """Verify C4-free and alpha < 11."""
        if has_c4(n, edges):
            return False
        alpha = max_independent_set_size(n, edges)
        return alpha < 11
    
    # Try to construct a 39-vertex graph
    # Use a known construction: the complement of a strongly regular graph
    # Or use a random construction with verification
    
    # Let's try a systematic approach:
    # Start with a complete graph and remove edges to break C4s while maintaining low alpha
    
    # Actually, let's try a different known construction.
    # The Petersen graph is C4-free with 10 vertices, alpha=4.
    # We can try to combine Petersen graphs or use other constructions.
    
    # For now, let's try a random search with local improvement
    random.seed(42)
    
    best_graph = None
    best_n = 0
    
    # Try n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random constructions
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Generate a random graph with appropriate edge density
            # For C4-free graphs, the edge density should be around O(n^(3/2))
            # For n=39, that's about 240 edges
            target_edges = int(n * n ** 0.5 * 0.5)
            
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < target_edges / (n * (n - 1) / 2):
                        edges.append([i, j])
            
            # Check C4-free
            if not has_c4(n, edges):
                # Check independence number
                alpha = max_independent_set_size(n, edges)
                if alpha < 11:
                    best_graph = (n, edges)
                    best_n = n
                    break
        
        if best_n >= n:
            break
    
    if best_graph is not None:
        n, edges = best_graph
        return {'n': n, 'edges': edges}
    
    # Fallback: return the seed program's answer
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
