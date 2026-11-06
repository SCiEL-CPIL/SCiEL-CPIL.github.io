import numpy as np
import time
import random

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
    n_tris = len(tri_indices)
    
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
        return int(np.max(sums))
    
    def try_swap(adj, u, v, x, y):
        """Apply edge swap: remove (u,v), add (x,y). Returns True if valid."""
        if adj[u, v] != 1:
            return False
        if adj[x, y] == 1:
            return False
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        return True
    
    def revert_swap(adj, u, v, x, y):
        adj[u, v] = 1
        adj[v, u] = 1
        adj[x, y] = 0
        adj[y, x] = 0
    
    current_adj = adj.copy()
    current_score = get_score(current_adj)
    best_adj = current_adj.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = 120.0  # Use full budget with safety margin
    
    # Precompute all possible edge pairs for fast move generation
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 8.0
    cooling = 0.9985
    min_T = 0.05
    steps = 0
    
    while time.time() - start_time < deadline - 15:  # Leave 15s for polish
        # Generate a move: swap an edge
        # Biased selection: prefer edges incident to high-degree vertices
        deg = current_adj.sum(axis=1)
        
        # Weighted random edge removal
        edge_weights = np.zeros(n*(n-1)//2, dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if current_adj[i, j] == 1:
                    edge_weights[idx] = 1.0 + 0.1 * (deg[i] + deg[j])
                else:
                    edge_weights[idx] = 0.0
                idx += 1
        
        # Find valid edges
        valid_edge_mask = edge_weights > 0
        if not np.any(valid_edge_mask):
            break
        
        valid_indices = np.where(valid_edge_mask)[0]
        weights = edge_weights[valid_indices]
        weights /= weights.sum()
        e_idx = np.random.choice(valid_indices, p=weights)
        
        # Convert flat index to (u, v)
        u = int(np.floor((1 + np.sqrt(1 + 8*e_idx)) / 2)) - 1
        v = int(e_idx - u*(u+1)//2)
        
        # Find a random non-edge to add, biased toward vertices that can help
        while True:
            x = np.random.randint(n)
            y = np.random.randint(n)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if current_adj[x, y] == 1:
                continue
            break
        
        # Apply move
        if not try_swap(current_adj, u, v, x, y):
            continue
        
        new_score = get_score(current_adj)
        steps += 1
        
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
                    revert_swap(current_adj, u, v, x, y)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final local search (Hill Climbing) on best_adj - more thorough
    improved = True
    polish_deadline = start_time + deadline - 2
    while improved and time.time() < polish_deadline:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u, v] == 1:
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        for y in range(x+1, n):
                            if y == u or y == v:
                                continue
                            if best_adj[x, y] == 0:
                                # Try swap
                                if try_swap(best_adj, u, v, x, y):
                                    new_score = get_score(best_adj)
                                    if new_score < best_score:
                                        best_score = new_score
                                        improved = True
                                    else:
                                        # Revert
                                        revert_swap(best_adj, u, v, x, y)
                                if time.time() > polish_deadline:
                                    break
                        if time.time() > polish_deadline:
                            break
                    if time.time() > polish_deadline:
                        break
            if time.time() > polish_deadline:
                break
    
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
                
    return {'n': n, 'edges': edges_list}
