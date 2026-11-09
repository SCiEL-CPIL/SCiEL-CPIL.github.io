import random
import itertools
import time

def construct():
    def popcount(x):
        return x.bit_count()

    def find_violation(n, adj):
        # Returns a mask of a regular induced subgraph of size >= 6, or 0 if none
        # Check sizes 6, 7, ..., n
        # Optimization: If no size 6, check 7, etc.
        # For n=21, C(21,6) is ~54k. C(21,7) is ~116k.
        # Total is ~2^21.
        # We can just iterate all masks? No, that's 2M masks.
        # Iterating combinations is better for small k.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degree of first vertex in subset
                first_v = subset[0]
                target_deg = popcount(adj[first_v] & mask)
                
                is_reg = True
                for v in subset[1:]:
                    if popcount(adj[v] & mask) != target_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return mask
        return 0

    def flip_edge(adj, u, v):
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def solve_for_n(n, max_iters=2000):
        # Initialize random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        for i in range(max_iters):
            mask = find_violation(n, adj)
            if mask == 0:
                # Extract edges
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return edges
            
            # Get vertices in violation
            verts = [v for v in range(n) if (mask >> v) & 1]
            
            # Pick two random vertices from the violation to flip an edge between them
            # This guarantees breaking the regularity of this specific subset
            u, v = random.sample(verts, 2)
            flip_edge(adj, u, v)
            
            # Optional: Sometimes flip an edge involving a vertex in violation and one outside?
            # No, that doesn't change degrees within the subset, so it doesn't break THIS violation.
            # It might prevent new ones, but let's stick to internal flips for guaranteed progress on current violation.
            
        return None

    # Try to find a valid graph for n=21, then 20, etc.
    for n in [21, 20, 19, 18]:
        # Try multiple restarts
        for attempt in range(10):
            edges = solve_for_n(n, max_iters=1000)
            if edges is not None:
                return {'n': n, 'edges': edges}
                
    # Fallback
    return {'n': 5, 'edges': []}
