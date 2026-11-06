import numpy as np
import time
import random

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    adj is a 31x31 numpy array of 0s and 1s.
    """
    # A^2
    A2 = adj @ adj
    # Sum over i < j of (A2[i,j]^2 - A2[i,j]) / 2
    # Mask for i < j
    mask = np.triu(np.ones((31, 31), dtype=bool), k=1)
    vals = A2[mask]
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_edges = None
    best_score = float('inf')
    
    # Initial random graphs
    for seed in range(1000):
        if time.time() > deadline:
            break
        random.seed(seed)
        # Generate random graph
        # Use a set to ensure distinct edges
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        
        # Convert to adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        score = count_c4(adj)
        
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            
    # If we have time, do local search on the best graph
    if best_edges is not None and time.time() < deadline:
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in best_edges:
            adj[u, v] = 1
            adj[v, u] = 1
        edge_set = set(best_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping edges
            # Pick a random edge to remove
            e_remove = random.choice(list(edge_set))
            u1, v1 = e_remove
            
            # Find a candidate edge to add
            # Try random pairs
            for _ in range(50):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                e_add = (u2, v2)
                
                if e_add in edge_set:
                    continue
                
                # Apply swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_score = count_c4(adj)
                
                if new_score < best_score:
                    best_score = new_score
                    edge_set.remove(e_remove)
                    edge_set.add(e_add)
                    best_edges = list(edge_set)
                    improved = True
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                
                if time.time() > deadline:
                    break
            
    return [[u, v] for u, v in best_edges]
