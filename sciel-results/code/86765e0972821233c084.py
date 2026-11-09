import random
import itertools
import time

def construct():
    def popcount(x):
        return bin(x).count("1")

    def is_valid_graph(n, edges):
        # Build bitset adjacency
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Precompute masks for all subsets? No, too many.
        # Iterate subsets of size 6, 7, ...
        # Optimization: If a subset of size k is regular, 
        # it doesn't necessarily mean a subset of size k-1 is.
        # But if we find ANY regular subset of size >= 6, it's invalid.
        
        # We can check size 6 first. If none, check 7, etc.
        # For n=20, C(20,6) = 38760. C(20,7) = 77520.
        # Total subsets >= 6 is ~ 2^20 = 1,048,576.
        # 1M subsets * 20 vertices * popcount (fast) 
        # In Python, this might still be tight for 120s if we do it naively.
        # Let's optimize the inner loop.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # Get degree of each vertex in subset
                # deg(u) = popcount(adj[u] & mask)
                # We need all degs to be equal.
                
                # Early exit: 
                # Calculate deg for first vertex
                first_v = subset[0]
                target_deg = popcount(adj[first_v] & mask)
                
                valid = True
                for v in subset[1:]:
                    if popcount(adj[v] & mask) != target_deg:
                        valid = False
                        break
                
                if valid:
                    return False
        return True

    def generate_random_graph(n, p):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    edges.append([u, v])
        return edges

    # Try to find a valid graph for decreasing n
    for n in range(20, 4, -1):
        # Try different probabilities
        for p in [0.5, 0.4, 0.6, 0.3, 0.7]:
            for _ in range(100): # 100 attempts per p
                edges = generate_random_graph(n, p)
                if is_valid_graph(n, edges):
                    return {'n': n, 'edges': edges}
    
    # Fallback to a small known valid graph if all else fails
    # n=5 is trivially valid (no subset of size 6)
    return {'n': 5, 'edges': []}
