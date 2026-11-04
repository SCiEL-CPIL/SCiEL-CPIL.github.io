"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 90.
    # We use a simple schoolbook decomposition which is always correct.
    # For each (i, j, k) in (0..n-1, 0..m-1, 0..p-1):
    # C[i,k] += A[i,j] * B[j,k]
    # U index for A[i,j] is i*m + j
    # V index for B[j,k] is j*p + k
    # W index for C[i,k] is k*n + i (trace convention)
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    # Tensor T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]B[j,k]
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k
    
    # Check a few random points to be safe, or all points if fast enough
    # n=3, m=5, p=6 -> 90 points. Very fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-9:
                    # This should never happen for schoolbook
                    # If it does, we have a bug in index calculation
                    pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
