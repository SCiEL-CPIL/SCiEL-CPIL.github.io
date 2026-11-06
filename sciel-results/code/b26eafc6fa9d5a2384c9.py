import time
import random
import numpy as np

def _is_k33_free(adj):
    """Check if graph is K3,3-free using bitmask common-neighbor check."""
    n = adj.shape[0]
    masks = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        m = 0
        for j in range(n):
            if adj[i, j]:
                m |= (1 << j)
        masks[i] = m
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                common = masks[a] & masks[b] & masks[c]
                if bin(common).count('1') >= 3:
                    return False
    return True

def _edges_from_adj(adj):
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n = 20
    best_edges = 0
    best_adj = np.zeros((n, n), dtype=bool)
    
    random.seed(42)
    
    # Try multiple random starts with different densities
    for attempt in range(30):
        if time.time() > deadline - 25:
            break
        
        p = 0.3 + 0.01 * attempt
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Greedy edge addition (try all missing edges)
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline - 20:
                    break
                if not adj[i, j]:
                    adj[i, j] = True
                    adj[j, i] = True
                    if not _is_k33_free(adj):
                        adj[i, j] = False
                        adj[j, i] = False
        
        # Local search: try to add more edges
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i, j]:
                        adj[i, j] = True
                        adj[j, i] = True
                        if _is_k33_free(adj):
                            added = True
                        else:
                            adj[i, j] = False
                            adj[j, i] = False
                        if added:
                            break
                if added:
                    break
            if not added:
                break
        
        # Edge swap local search
        for _ in range(300):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try random edge addition
            for _ in range(30):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j or adj[i, j]:
                    continue
                adj[i, j] = True
                adj[j, i] = True
                if _is_k33_free(adj):
                    improved = True
                    break
                else:
                    adj[i, j] = False
                    adj[j, i] = False
            
            if not improved:
                # Try swap: remove a random edge, add a different one
                edge_list = _edges_from_adj(adj)
                if len(edge_list) < 1:
                    break
                e1 = random.choice(edge_list)
                adj[e1[0], e1[1]] = False
                adj[e1[1], e1[0]] = False
                for _ in range(30):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j or adj[i, j]:
                        continue
                    adj[i, j] = True
                    adj[j, i] = True
                    if _is_k33_free(adj):
                        # Try to re-add e1
                        adj[e1[0], e1[1]] = True
                        adj[e1[1], e1[0]] = True
                        if _is_k33_free(adj):
                            improved = True
                        else:
                            adj[e1[0], e1[1]] = False
                            adj[e1[1], e1[0]] = False
                        break
                    else:
                        adj[i, j] = False
                        adj[j, i] = False
                if not improved:
                    # Restore e1
                    adj[e1[0], e1[1]] = True
                    adj[e1[1], e1[0]] = True
            
            if not improved:
                break
        
        edge_count = int(np.sum(adj)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
    
    return _edges_from_adj(best_adj)
