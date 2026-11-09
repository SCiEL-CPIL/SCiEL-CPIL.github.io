"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def check_decomp(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def to_list(U, V, W, r):
        return {
            "U": [[float(U[i, c]) for c in range(r)] for i in range(n * m)],
            "V": [[float(V[j, c]) for c in range(r)] for j in range(m * p)],
            "W": [[float(W[k, c]) for c in range(r)] for k in range(p * n)]
        }
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return to_list(U, V, W, rank)
    
    # Try a structured decomposition
    # For C[i,k] = sum_j A[i,j]*B[j,k], use the fact that we can group by i
    # For each i, C[i,:] = A[i,:] @ B, which is a 1x3 times 3x5 = 1x5
    
    # One approach: for each i, use a rank-3 decomposition of the 1x3 @ 3x5 multiplication
    # But we can share across i
    
    # Let's try: for each i, C[i,k] = sum_j A[i,j]*B[j,k]
    # Use 3 terms per i for the j-sum, but try to share
    
    # Actually, let's just try to find a valid decomposition with rank ~20-25
    # by using a local search from a good initialization
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 20 to 28
    for r in range(20, 29):
        if time.time() > deadline - 10:
            break
        
        # Initialize with a structured pattern
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Pattern: for each i, use r/3 columns for the j-sum
        cols_per_i = r // n
        extra = r % n
        
        col = 0
        for i in range(n):
            c_count = cols_per_i + (1 if i < extra else 0)
            for c in range(c_count):
                # For column col, handle part of the multiplication for row i
                for j in range(m):
                    U[i*m+j, col] = 1.0
                for k in range(p):
                    W[k*n+i, col] = 1.0
                # V: distribute across j
                for j in range(m):
                    V[j*p+k, col] = 1.0 / m if False else 1.0  # placeholder
                col += 1
        
        # This isn't quite right. Let me just do a proper random search with more attempts.
        
        found = False
        for attempt in range(500):
            if time.time() > deadline - 2:
                break
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.2:
                        U[idx, c] = random.choice([1, -1, 0.5, -0.5])
                for idx in range(m * p):
                    if random.random() < 0.2:
                        V[idx, c] = random.choice([1, -1, 0.5, -0.5])
                for idx in range(p * n):
                    if random.random() < 0.2:
                        W[idx, c] = random.choice([1, -1, 0.5, -0.5])
            
            if check_decomp(U, V, W, r):
                best = to_list(U, V, W, r)
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
