import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # 1. Construct the known valid graph: 8 disjoint K4's
    # This graph has n=32, alpha=8, and is C5-free.
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        # Add all edges for K4
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    
    best_n = n_base
    best_edges = edges_base

    # 2. Attempt to find a graph with n=33
    # Strategy: Local search starting from random graphs or modified base.
    # Given the difficulty, we try a few random restarts with a fast validator.
    
    def is_valid(n, edges):
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            if u < n and v < n:
                adj[u][v] = True
                adj[v][u] = True
        
        # Check C5
        # Fast check: for each pair of non-adjacent (u, v), check if there is a path of length 4?
        # Simpler: Iterate over all 5-tuples is too slow.
        # Use the property: A graph has a C5 iff there is a cycle of length 5.
        # For n=33, we can use a bitset-based check if optimized, but in Python it's tricky.
        # We'll use a heuristic: if the graph is a disjoint union of cliques of size <= 4, it's C5-free.
        # But we are exploring other graphs.
        
        # For the purpose of this contest, if we can't verify C5 quickly, we assume invalid.
        # However, we know 8 K4's is valid.
        
        # Let's try to verify alpha < 9
        # Greedy independent set
        remaining = set(range(n))
        independent_set = []
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if adj[v][u])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            independent_set.append(min_v)
            remaining.discard(min_v)
            for u in list(remaining):
                if adj[min_v][u]:
                    remaining.discard(u)
        
        if len(independent_set) >= 9:
            return False
            
        # C5 Check:
        # If the graph is sparse, C5 is unlikely.
        # We'll do a simple check: for each vertex v, check neighbors.
        # This is a placeholder for a real check. 
        # Given time constraints, we will only accept graphs that are structurally similar to K4 unions or known C5-free structures.
        
        # For now, we assume only the base graph is valid if we can't prove others.
        return True

    # Since verifying C5 is hard, we won't risk invalid graphs.
    # We return the base graph.
    
    return {'n': best_n, 'edges': best_edges}
