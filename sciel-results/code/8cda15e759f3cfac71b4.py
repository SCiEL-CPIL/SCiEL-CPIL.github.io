import numpy as np
import time
import random

def construct():
    n = 40
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        # Adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a walk of length 5
            # Compute e_u^T A^5
            # v_vec starts as e_u
            v_vec = np.zeros(n, dtype=np.int64)
            v_vec[u] = 1
            
            # Multiply by A 5 times
            # Opt: Use matrix multiplication
            for _ in range(5):
                v_vec = v_vec @ A
            
            if v_vec[v] == 0:
                # Safe to add
                A[u, v] = 1
                A[v, u] = 1
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        # If we reached a high number, we might break early?
        # No, keep searching for better.
        
    return [[u, v] for u, v in best_edges]
