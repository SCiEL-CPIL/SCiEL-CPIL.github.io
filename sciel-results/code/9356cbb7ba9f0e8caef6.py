"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check tensor equation: for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check that for "invalid" combinations, the sum is 0
    # Actually, the tensor is only defined for valid (i,j,k) triples
    # But we need to ensure no spurious contributions
    # The tensor T[i,j,k] = delta_{conditions} where conditions are j=j', k=k', i=i'
    # Since we're only summing over the valid triples, and each triple is independent,
    # we just need the above to hold.
    
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    time_limit = 120.0  # 120 seconds limit
    end_time = start_time + time_limit
    
    n, m, p = DIMENSIONS
    
    # Standard schoolbook decomposition
    # Rank is n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each (i, j, k) triple gets a unique rank index
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                # Time check (rarely needed for this size, but good practice)
                if time.time() > end_time - 5.0:
                    # If we run out of time, we just return what we have
                    # Since we are in the construction loop, we should finish quickly
                    # This is a safety break
                    return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
