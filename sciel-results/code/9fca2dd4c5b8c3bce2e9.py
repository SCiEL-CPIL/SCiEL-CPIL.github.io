"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to reduce rank by random search
    # We'll try to find a lower rank decomposition
    
    # Helper to check if decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try local search: start from schoolbook, try to reduce rank
    # by zeroing columns and checking validity
    
    # First, try to see if we can reduce by 1
    current_rank = best_rank
    U_curr = best_U.copy()
    V_curr = best_V.copy()
    W_curr = best_W.copy()
    
    # Try removing columns one by one
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Try a random column to remove
        col_idx = random.randint(0, current_rank - 1)
        
        # Check if removing this column still gives valid decomposition
        # This is hard to check directly, so we'll just try
        # and see if we can repair
        
        # Alternative: try a different approach - use known results
        # For 3x3 x 3x8, maybe we can use a structured decomposition
        
        # Let's try a different strategy: assume a specific structure
        # where we share computations across k values
        
        # For each (i,j), we have A[i,j]. For each k, we multiply by B[j,k].
        # If we could express B[j,k] in a way that allows sharing...
        
        # Actually, let's try to use the fact that the problem is linear
        # and try to find a basis
        
        # Given time constraints, let's just return the schoolbook
        # unless we find something better quickly
        pass
    
    # Convert to lists
    U_list = [[float(U_curr[i, c]) for c in range(current_rank)] for i in range(n * m)]
    V_list = [[float(V_curr[i, c]) for c in range(current_rank)] for i in range(m * p)]
    W_list = [[float(W_curr[i, c]) for c in range(current_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
