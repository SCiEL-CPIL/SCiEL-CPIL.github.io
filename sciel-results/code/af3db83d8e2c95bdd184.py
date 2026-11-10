import numpy as np
import random
import time

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140  # Leave 10s margin
    
    # Initialize a random graph with m edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = set(random.sample(all_edges, m))
    
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    def compute_S(A):
        A2 = A @ A
        return int(np.sum(A2 * A2))
    
    current_S = compute_S(A)
    best_S = current_S
    best_A = A.copy()
    
    T = 50.0  # Higher initial temperature for exploration
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge to remove
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and A[u, v] == 1:
                break
        
        # Pick random non-edge to add
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x != y and A[x, y] == 0:
                break
        
        # Apply move
        A[u, v] = 0
        A[v, u] = 0
        A[x, y] = 1
        A[y, x] = 1
        
        new_S = compute_S(A)
        delta_S = new_S - current_S
        
        if delta_S < 0 or (T > 1e-6 and random.random() < np.exp(-delta_S / T)):
            current_S = new_S
            if current_S < best_S:
                best_S = current_S
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 0
            A[y, x] = 0
            
        T *= cooling
        
    # Convert best_A to list of lists
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                result.append([i, j])
                
    return result
