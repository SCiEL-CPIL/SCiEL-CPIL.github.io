import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for integer/half-integer entries
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verification Step:
    # Check if sum_c U[:,c] V[:,c] == T and W matches the trace convention
    # T[i,j,k] = 1 if i,j,k are valid indices for the multiplication
    # The tensor T for matrix multiplication C = AB is defined such that
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The decomposition is C_{ik} = sum_c (sum_j U_{ij,c} V_{jk,c}) W_{ik,c} ?
    # No, the standard decomposition is:
    # T_{ijk} = sum_c U_{ic,c} V_{jc,c} W_{kc,c} ?
    # Let's check the interface:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Usually:
    # U_{ij, c}, V_{jk, c}, W_{ki, c} ?
    # The prompt says: W[k*n+i, r]. This is W_{ki, r}.
    # So the tensor entry T_{i,j,k} should be reconstructed as:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook:
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    # Product is 1. Sum is 1.
    # This matches the matrix multiplication tensor where T_{ijk}=1 for all valid i,j,k.
    
    # We can skip heavy verification to save time, but a light check is good.
    # Since we constructed it correctly, we return it.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
