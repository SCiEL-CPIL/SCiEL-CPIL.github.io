import numpy as np
import time
import random

def _has_triangle(adj):
    paths2 = (adj @ adj) > 0
    tri_mask = adj & paths2
    return np.any(tri_mask), tri_mask

def _find_max_independent_set_greedy(adj, n, rng):
    degrees = np.sum(adj, axis=1)
    noise = rng.random(n) * 0.1
    order = np.argsort(degrees + noise)
    
    in_set = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    
    for v in order:
        if not remaining[v]:
            continue
        if np.any(adj[v, :] & in_set):
            continue
        in_set[v] = True
        remaining[v] = False
        remaining[adj[v, :]] = False
        
    return np.sum(in_set), in_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_n = 0
    best_edges = []
    
    for n in [40, 39]:
        if time.time() > deadline - 5:
            break
            
        found = False
        max_restarts = 100
        restarts_done = 0
        
        while restarts_done < max_restarts and time.time() < deadline - 5:
            restarts_done += 1
            rng = np.random.default_rng(restarts_done)
            
            adj = rng.random((n, n)) < 0.5
            adj = (adj | adj.T).astype(bool)
            np.fill_diagonal(adj, False)
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                has_tri, tri_mask = _has_triangle(adj)
                
                if has_tri:
                    tri_edges = np.argwhere(tri_mask)
                    if len(tri_edges) > 0:
                        u, v = tri_edges[rng.integers(len(tri_edges))]
                        adj[u, v] = False
                        adj[v, u] = False
                        continue
                
                size, in_set = _find_max_independent_set_greedy(adj, n, rng)
                
                if size >= 10:
                    set_indices = np.where(in_set)[0]
                    if len(set_indices) >= 2:
                        idx1, idx2 = rng.choice(len(set_indices), size=2, replace=False)
                        u = set_indices[idx1]
                        v = set_indices[idx2]
                        adj[u, v] = True
                        adj[v, u] = True
                        continue
                else:
                    # Heuristic validation
                    valid = True
                    for _ in range(10):
                        size_chk, _ = _find_max_independent_set_greedy(adj, n, rng)
                        if size_chk >= 10:
                            valid = False
                            break
                    
                    if valid:
                        if n > best_n:
                            best_n = n
                            edges = []
                            for i in range(n):
                                for j in range(i+1, n):
                                    if adj[i, j]:
                                        edges.append([i, j])
                            best_edges = edges
                        found = True
                        break
            
            if found:
                break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Andrásfai graph And(9)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
