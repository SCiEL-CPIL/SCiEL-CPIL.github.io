import numpy as np
import time
import random

def construct():
    n = 20
    # A is the adjacency matrix
    A = np.zeros((n, n), dtype=int)
    eye = np.eye(n, dtype=bool)
    mask = ~eye
    
    def is_valid(A):
        C = A @ A
        return np.all(C[mask] <= 2)
        
    def edge_count(A):
        return int(np.sum(A) // 2)
        
    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    edges.append([i, j])
        return edges

    # Initial greedy construction
    # Randomize order to get different starting points
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        A[u, v] = 1
        A[v, u] = 1
        if not is_valid(A):
            A[u, v] = 0
            A[v, u] = 0
            
    best_A = A.copy()
    best_count = edge_count(A)
    
    start_time = time.time()
    time_limit = 115.0
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try to add edges
        # Generate candidate non-edges
        # To speed up, only consider pairs with C_uv <= 1
        C = A @ A
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 0 and C[i, j] <= 1:
                    candidates.append((i, j))
                    
        if candidates:
            random.shuffle(candidates)
            for u, v in candidates:
                A[u, v] = 1
                A[v, u] = 1
                if is_valid(A):
                    improved = True
                    curr_count = edge_count(A)
                    if curr_count > best_count:
                        best_count = curr_count
                        best_A = A.copy()
                    break
                else:
                    A[u, v] = 0
                    A[v, u] = 0
        else:
            # No candidates with C_uv <= 1, meaning all non-edges have C_uv >= 2
            # We must remove an edge to proceed
            # Remove a random edge
            edges = get_edges(A)
            if edges:
                u, v = random.choice(edges)
                A[u, v] = 0
                A[v, u] = 0
            else:
                break
                
    return get_edges(best_A)
