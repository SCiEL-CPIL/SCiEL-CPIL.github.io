import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin from 150s limit
    
    # Schoolbook baseline: rank = n*m*p = 54
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Verify helper using NumPy for speed
    def verify(U, V, W, R):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U3 = U_arr.reshape(n, m, R)
        V3 = V_arr.reshape(m, p, R)
        W3 = W_arr.reshape(p, n, R)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # einsum indices: i j r, j k r, k i r -> i j k
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # Check if T is all ones
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Strategy: Try to find better decompositions using structured search
    # We focus on ranks slightly below schoolbook to see if a simple improvement exists.
    # Values from a small set to encourage integer/half-integer solutions.
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Try ranks from 45 to 53 (skipping 54 as it's the baseline)
    # We start from higher ranks because they are more likely to have a solution
    # if one exists near the boundary.
    for R in range(53, 44, -1):
        if time.time() > deadline - 10:
            break
        
        # Limit attempts to ensure we don't exceed time
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Fill with random values
            # Using random.choice in a loop is slow, so we use vectorized random choice
            # But the values are discrete.
            # We can generate random indices and map to values.
            U_indices = np.random.randint(0, len(values), size=(n * m, R))
            V_indices = np.random.randint(0, len(values), size=(m * p, R))
            W_indices = np.random.randint(0, len(values), size=(p * n, R))
            
            U = values[U_indices]
            V = values[V_indices]
            W = values[W_indices]
            
            # Check if this works
            if verify(U, V, W, R):
                best_rank = R
                best = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                return best
        
        # If we found a valid decomposition at this rank, return it
        # (The inner loop returns immediately upon finding one)
        
    # If no better decomposition found, return the schoolbook baseline
    return best
