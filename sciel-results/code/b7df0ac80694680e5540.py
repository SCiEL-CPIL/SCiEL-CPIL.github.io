import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 40
    m_target = 401
    
    # Initial graph: 4-partite cycle A-B-C-D-A
    # A: 0-9, B: 10-19, C: 20-29, D: 30-39
    # Edges: A-B, B-C, C-D, D-A
    adj = np.zeros((n, n), dtype=np.int8)
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        
    # Build base
    for a in range(10):
        for b in range(10, 20):
            add_edge(a, b)
    for b in range(10, 20):
        for c in range(20, 30):
            add_edge(b, c)
    for c in range(20, 30):
        for d in range(30, 40):
            add_edge(c, d)
    for d in range(30, 40):
        for a in range(10):
            add_edge(d, a)
            
    # Add 1 random edge to reach 401
    # Pick a random intra-part edge to start, or cross-part
    # Intra-part A
    u = random.randint(0, 9)
    v = random.randint(0, 9)
    while v == u:
        v = random.randint(0, 9)
    add_edge(u, v)
    
    def get_max_book():
        # Max over edges of common neighbors
        max_b = 0
        # Only need to check edges
        # adj is symmetric
        # Common neighbors of u,v = sum(adj[u] * adj[v])
        # We can vectorize this?
        # For each edge (u,v), compute dot product.
        # There are 401 edges.
        # 401 * 40 * 40 operations is ~ 640k. Very fast.
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
                    
        for u, v in edges:
            common = np.dot(adj[u], adj[v])
            if common > max_b:
                max_b = common
        return max_b
        
    current_score = get_max_book()
    best_score = current_score
    best_adj = adj.copy()
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    while time.time() < deadline:
        # Pick random edge to remove
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        while u == v or not adj[u, v]:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            
        # Pick random non-edge to add
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        while x == y or adj[x, y]:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            
        # Apply move
        remove_edge(u, v)
        add_edge(x, y)
        
        new_score = get_max_book()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Accept with probability
            if T > min_T:
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    remove_edge(x, y)
                    add_edge(u, v)
            else:
                # Revert
                remove_edge(x, y)
                add_edge(u, v)
                
        T *= cooling
        
    # Restore best
    adj = best_adj
    
    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                result.append([i, j])
                
    # Verify count
    if len(result) != m_target:
        # Fallback to base if something went wrong
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result

