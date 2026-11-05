import numpy as np
import time

def construct():
    n = 16
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    I = tri_indices[:, 0]
    J = tri_indices[:, 1]
    K = tri_indices[:, 2]
    
    # Initial graph: K_{8,8} + {0,1}
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def get_score(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return np.max(sums)
    
    current_adj = adj.copy()
    current_score = get_score(current_adj)
    best_adj = current_adj.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = 50.0 # 50 seconds budget for search
    
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < deadline:
        # Generate a move: swap an edge
        # Find a random edge to remove
        edges = np.argwhere(current_adj == 1)
        if len(edges) == 0: break
        e_idx = np.random.randint(len(edges))
        u, v = edges[e_idx]
        
        # Find a random non-edge to add
        while True:
            x = np.random.randint(n)
            y = np.random.randint(n)
            if x == y: continue
            if current_adj[x, y] == 1: continue
            break
        
        # Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1
        
        new_score = get_score(current_adj)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
            else:
                current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # Final local search (Hill Climbing) on best_adj
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u, v] == 1:
                    for x in range(n):
                        if x == u or x == v: continue
                        for y in range(x+1, n):
                            if y == u or y == v: continue
                            if best_adj[x, y] == 0:
                                # Try swap
                                best_adj[u, v] = 0
                                best_adj[v, u] = 0
                                best_adj[x, y] = 1
                                best_adj[y, x] = 1
                                
                                new_score = get_score(best_adj)
                                if new_score < best_score:
                                    best_score = new_score
                                    improved = True
                                else:
                                    # Revert
                                    best_adj[u, v] = 1
                                    best_adj[v, u] = 1
                                    best_adj[x, y] = 0
                                    best_adj[y, x] = 0
                                if time.time() - start_time > deadline - 5:
                                    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i, j] == 1]}
        
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
                
    return {'n': n, 'edges': edges_list}
