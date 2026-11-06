import numpy as np
import time
import random

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 110  # Safety margin
    
    # Initialize random graph
    # We can use a fixed seed for reproducibility or just random
    random.seed(42)
    np.random.seed(42)
    
    # Create a random adjacency matrix
    A = np.zeros((n, n), dtype=int)
    edges = []
    while len(edges) < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u < v and A[u, v] == 0:
            A[u, v] = 1
            A[v, u] = 1
            edges.append((u, v))
    
    def count_c4(A):
        # C4 count is sum_{i<j} binom(c_ij, 2)
        # c_ij is (A^2)_ij
        A2 = A @ A
        # We only care about i < j
        # binom(x, 2) = x(x-1)/2
        # We can compute sum over all i,j and divide by 2?
        # No, (i,j) and (j,i) are same.
        # Sum_{i != j} binom(A2[i,j], 2) / 2
        # A2[i,i] is degree, binom(deg, 2) is number of triangles? No, we want C4.
        # C4s do not involve loops.
        # So we sum over i < j.
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = A2[i, j]
                if c > 1:
                    c4_count += c * (c - 1) // 2
        return c4_count
    
    current_c4 = count_c4(A)
    best_c4 = current_c4
    best_A = A.copy()
    
    T = 1.0
    cooling_rate = 0.9999
    
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(edges)
        
        # Pick a random non-edge to add
        # We need to find a non-edge.
        # We can pick random u2, v2 until we find a non-edge.
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 < v2 and A[u2, v2] == 0:
                break
        
        # Swap
        A[u, v] = 0
        A[v, u] = 0
        A[u2, v2] = 1
        A[v2, u2] = 1
        
        new_c4 = count_c4(A)
        
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            current_c4 = new_c4
            # Update edges list
            edges.remove((u, v))
            edges.append((u2, v2))
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_A = A.copy()
        else:
            # Reject
            A[u, v] = 1
            A[v, u] = 1
            A[u2, v2] = 0
            A[v2, u2] = 0
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Extract edges from best_A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
