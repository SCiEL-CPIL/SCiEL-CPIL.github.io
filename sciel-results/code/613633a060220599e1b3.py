import numpy as np
import random

def construct():
    n = 20
    best_edges_count = -1
    best_edges = []
    
    # Use multiple random restarts
    for seed in range(100):
        random.seed(seed)
        
        # Initialize complete graph
        adj = np.ones((n, n), dtype=np.int8)
        np.fill_diagonal(adj, 0)
        
        # deg is a numpy array
        deg = np.sum(adj, axis=1).astype(np.int64)
        
        while True:
            # Find pair (u, v) with maximum codegree
            C = adj @ adj.T
            np.fill_diagonal(C, -1)
            
            max_codeg = np.max(C)
            
            if max_codeg <= 2:
                # Valid
                current_count = int(np.sum(adj)) // 2
                if current_count > best_edges_count:
                    best_edges_count = current_count
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
                break
                
            # Get all pairs with max codegree
            max_pairs = np.argwhere(C == max_codeg)
            if len(max_pairs) > 1:
                u, v = max_pairs[random.randint(0, len(max_pairs)-1)]
            else:
                u, v = max_pairs[0]
            
            # Ensure u < v for consistent edge representation if needed, 
            # but for removal it doesn't matter.
            
            # Get common neighbors
            commons_mask = adj[u] & adj[v]
            commons = np.where(commons_mask == 1)[0]
            
            # Pick a random common neighbor
            w = commons[random.randint(0, len(commons)-1)]
            
            # Heuristic selection based on seed
            if seed < 33:
                # Remove from higher degree
                if deg[u] >= deg[v]:
                    adj[u, w] = 0
                    adj[w, u] = 0
                    deg[u] -= 1
                    deg[w] -= 1
                else:
                    adj[v, w] = 0
                    adj[w, v] = 0
                    deg[v] -= 1
                    deg[w] -= 1
            elif seed < 66:
                # Remove from lower degree
                if deg[u] < deg[v]:
                    adj[u, w] = 0
                    adj[w, u] = 0
                    deg[u] -= 1
                    deg[w] -= 1
                else:
                    adj[v, w] = 0
                    adj[w, v] = 0
                    deg[v] -= 1
                    deg[w] -= 1
            else:
                # Random
                if random.random() < 0.5:
                    adj[u, w] = 0
                    adj[w, u] = 0
                    deg[u] -= 1
                    deg[w] -= 1
                else:
                    adj[v, w] = 0
                    adj[w, v] = 0
                    deg[v] -= 1
                    deg[w] -= 1

    return best_edges
