import time
import random
import numpy as np

def _creates_c4(adj, u, v):
    nu = adj[u]
    nv = adj[v]
    if not nu.any() or not nv.any():
        return False
    return adj[nu][:, nv].any()

def _construct_c4_free(n, min_deg_required, time_limit, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    start_time = time.time()
    
    # Try multiple restarts
    # Each restart: Initialize, then Local Search
    
    while time.time() - start_time < time_limit:
        # 1. Initialization
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Randomized Greedy Initialization
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if not _creates_c4(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
        
        # 2. Local Search
        # We will iterate until we find a solution or get stuck for a while
        stuck_count = 0
        max_stuck = 2000
        
        while time.time() - start_time < time_limit:
            min_deg = int(np.min(degrees))
            if min_deg >= min_deg_required:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # Find vertex with min degree
            min_deg_vertex = int(np.argmin(degrees))
            
            # Try to add edge to min_deg_vertex
            non_neighbors = np.where(~adj[min_deg_vertex])[0]
            non_neighbors = non_neighbors[non_neighbors != min_deg_vertex]
            
            if len(non_neighbors) > 0:
                nn_list = non_neighbors.tolist()
                rng.shuffle(nn_list)
                
                found = False
                for v in nn_list:
                    if not _creates_c4(adj, min_deg_vertex, v):
                        adj[min_deg_vertex, v] = True
                        adj[v, min_deg_vertex] = True
                        degrees[min_deg_vertex] += 1
                        degrees[v] += 1
                        found = True
                        stuck_count = 0
                        break
                
                if found:
                    continue
            
            # Stuck: Remove a random edge
            edge_indices = np.argwhere(adj)
            edge_indices = edge_indices[edge_indices[:, 0] < edge_indices[:, 1]]
            
            if len(edge_indices) == 0:
                break
                
            idx = rng.randint(0, len(edge_indices) - 1)
            u, v = edge_indices[idx]
            adj[u, v] = False
            adj[v, u] = False
            degrees[u] -= 1
            degrees[v] -= 1
            
            stuck_count += 1
            if stuck_count > max_stuck:
                break

    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Try n=47
    t_limit_47 = 100.0
    result = _construct_c4_free(47, 8, t_limit_47, seed=123)
    if result:
        return result
        
    # Try n=46
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 10:
        result = _construct_c4_free(46, 7, remaining - 5, seed=456)
        if result:
            return result
            
    # Fallback
    return {'n': 39, 'edges': []}
