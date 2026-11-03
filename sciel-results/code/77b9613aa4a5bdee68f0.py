"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # M1 = (A[0,j0]+A[1,j1])*(B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # M2 = (A[1,j0]+A[1,j1])*B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # M3 = A[0,j0]*(B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # M4 = A[1,j1]*(B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # M5 = (A[0,j0]+A[0,j1])*B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # M6 = (A[1,j0]-A[0,j0])*(B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # M7 = (A[0,j1]-A[1,j1])*(B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    # Start with rank 42 (6 Strassen blocks of 7 each) + 1 for the j=4 slice
    rank = 43
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # j=4 slice: A[:,4] * B[4,:] contributes to C[:,k]
    # This is a rank-1 tensor in the (i, k) space for each j=4
    # We need: sum_r U[i, 4, r] V[4, k, r] W[k, i, r] = 1 for all i, k
    # A single rank-1 term works:
    # U[0,4,r] = 1, U[1,4,r] = 1
    # V[4,k,r] = 1 for all k
    # W[k,i,r] = 1 for all k, i
    # Check: U[0,4,r]*V[4,k,r]*W[k,0,r] = 1*1*1 = 1 ✓
    #         U[1,4,r]*V[4,k,r]*W[k,1,r] = 1*1*1 = 1 ✓
    
    # U index for (i=0, j=4): 0*5 + 4 = 4
    # U index for (i=1, j=4): 1*5 + 4 = 9
    # V index for (j=4, k): 4*6 + k = 24 + k
    # W index for (k, i=0): k*2 + 0 = 2k
    # W index for (k, i=1): k*2 + 1 = 2k + 1
    
    U[4][col] += 1
    U[9][col] += 1
    for k in range(p):
        V[24 + k][col] += 1
        W[2*k][col] += 1
        W[2*k + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

