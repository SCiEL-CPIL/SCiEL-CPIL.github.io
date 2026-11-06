import numpy as np
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_edges = []
    best_count = 0
    
    def is_c6_free(adj):
        # adj is a numpy array of 0s and 1s
        # Compute A^5
        # A^2 = A @ A
        # A^4 = A^2 @ A^2
        # A^5 = A^4 @ A
        A2 = adj @ adj
        A4 = A2 @ A2
        A5 = A4 @ adj
        
        # Check all edges (u, v) where adj[u, v] == 1
        # If A5[u, v] > 0, there is a path of length 5, so adding/keeping edge (u,v) forms a C6
        # Note: adj is symmetric
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    if A5[u, v] > 0:
                        return False, (u, v)
        return True, None

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    edges.append([u, v])
        return edges

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def local_search(adj, time_limit):
        # Prune phase
        while time.time() < time_limit:
            is_free, edge_in_c6 = is_c6_free(adj)
            if is_free:
                break
            if edge_in_c6:
                u, v = edge_in_c6
                adj[u, v] = 0
                adj[v, u] = 0
            else:
                break
        
        # Add phase
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle order of missing edges to avoid bias
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        missing.append((u, v))
            
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() >= time_limit:
                    break
                
                # Try adding edge (u, v)
                adj[u, v] = 1
                adj[v, u] = 1
                
                is_free, _ = is_c6_free(adj)
                
                if is_free:
                    improved = True
                    # Keep the edge
                else:
                    # Revert
                    adj[u, v] = 0
                    adj[v, u] = 0
        
        return adj

    # Try multiple random starts
    # Densities to try: 0.5, 0.6, 0.7, 0.8
    densities = [0.7, 0.6, 0.8, 0.5, 0.75]
    
    for density in densities:
        if time.time() > deadline:
            break
            
        # Run local search a few times for this density
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            adj = (np.random.rand(n, n) < density).astype(int)
            np.fill_diagonal(adj, 0)
            # Symmetrize
            adj = np.maximum(adj, adj.T)
            
            # Local search
            adj = local_search(adj, deadline)
            
            # Count edges
            count = count_edges(adj)
            if count > best_count:
                best_count = count
                best_edges = get_edges_from_adj(adj)
                
    return best_edges
