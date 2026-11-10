import time
import random
import numpy as np

def _is_k33_free(adj):
    """Check if graph with adjacency bitmasks is K3,3-free.
    adj: list of 20 integers, each a bitmask of neighbors.
    K3,3 exists iff some triple of vertices has >=3 common neighbors.
    """
    n = 20
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                common = adj[i] & adj[j] & adj[k]
                if bin(common).count('1') >= 3:
                    return False
    return True

def _edge_count(adj):
    return sum(bin(a).count('1') for a in adj) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 20
    best_adj = None
    best_edges = 0
    
    # Try multiple random starts with local search
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initial graph
        p = 0.35
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search: greedy edge addition
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        continue
                    # Try adding edge (i,j)
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    if _is_k33_free(adj):
                        improved = True
                    else:
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
        
        # Local search: edge swaps (remove one, add another)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Collect existing and missing edges
            existing = []
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        existing.append((i,j))
                    else:
                        missing.append((i,j))
            
            for (i1,j1) in existing:
                if time.time() > deadline:
                    break
                for (i2,j2) in missing:
                    if (i1,j1) == (i2,j2):
                        continue
                    # Remove (i1,j1), add (i2,j2)
                    adj[i1] ^= (1 << j1)
                    adj[j1] ^= (1 << i1)
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
                    if _is_k33_free(adj):
                        improved = True
                        break
                    else:
                        adj[i2] ^= (1 << j2)
                        adj[j2] ^= (1 << i2)
                        adj[i1] |= (1 << j1)
                        adj[j1] |= (1 << i1)
                if improved:
                    break
        
        ec = _edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for i in range(20):
        for j in range(i+1, 20):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
