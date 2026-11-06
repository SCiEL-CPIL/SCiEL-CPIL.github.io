"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # 130s safety margin
    
    def has_c4_fast(adj, n):
        """Fast C4 check using numpy."""
        common = adj @ adj.T
        np.fill_diagonal(common, 0)
        return np.any(common >= 2)
    
    def min_degree_fast(adj, n):
        return np.min(adj.sum(axis=1))
    
    def try_random(n, min_deg, max_attempts=5000, seed=42):
        """Try random graphs to find C4-free with min degree >= min_deg."""
        random.seed(seed)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            p = min_deg / n * 1.2
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            if min_degree_fast(adj, n) < min_deg:
                continue
            
            if has_c4_fast(adj, n):
                continue
            
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        return None
    
    # Try n=47 first, then decreasing
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result = try_random(n, min_deg)
        if result is not None:
            return result
    
    # Fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
