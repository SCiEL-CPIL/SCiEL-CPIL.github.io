import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def add_edge_safe(adj, n, u, v):
        """Check if adding edge (u,v) creates a C4. Return True if safe."""
        # u and v must have at most 1 common neighbor
        common = np.minimum(adj[u], adj[v])
        common[u] = 0
        common[v] = 0
        return common.sum() <= 1
    
    def try_n(n):
        target_min_deg = n - 39
        if target_min_deg < 1:
            target_min_deg = 1
        
        best_adj = None
        best_min_deg = -1
        
        for attempt in range(500):
            if elapsed() > deadline - 5:
                break
            
            # Start with empty graph, add edges greedily in random order
            adj = np.zeros((n, n), dtype=np.int8)
            
            # Create edge list in random order
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # Add edges if they don't create C4
            for u, v in edges:
                if elapsed() > deadline - 5:
                    break
                if add_edge_safe(adj, n, u, v):
                    adj[u][v] = 1
                    adj[v][u] = 1
            
            # Check minimum degree
            degrees = adj.sum(axis=1)
            md = int(degrees.min())
            
            if md > best_min_deg:
                best_min_deg = md
                best_adj = adj.copy()
            
            if md >= target_min_deg:
                edge_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edge_list.append([i, j])
                return n, edge_list
        
        if best_adj is not None and best_min_deg >= target_min_deg:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if best_adj[i][j]:
                        edge_list.append([i, j])
            return n, edge_list
        
        return None
    
    # Try from largest n down
    for n in [47, 46, 45, 44, 43]:
        result = try_n(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}
    
    # Fallback
    return {'n': 39, 'edges': []}
