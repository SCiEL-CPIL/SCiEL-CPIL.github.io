"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split the 3x8 matrix B into four 3x2 blocks (B1, B2, B3, B4).
    For each 3x2 block, split into a 2x2 part (rows 0,1) and a 1x2 part (row 2).
    The 2x2 * 2x2 part uses Strassen (rank 7).
    The 2x1 * 1x2 part uses schoolbook (rank 4).
    Total: 4 * (7 + 4) = 44.
    
    W convention: W[k*n + i, r] where k is output column, i is output row.
    U convention: U[i*m + j, r] where i is A row, j is A column.
    V convention: V[j*p + k, r] where j is B row, k is B column.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A elements: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B elements: B[0,k0]=V[k0], B[0,k1]=V[k1], B[1,k0]=V[p+k0], B[1,k1]=V[p+k1]
        # C elements: C[i,k] = W[k*n+i]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[k1][col] = 1; V[p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1
        V[p + k0][col] = 1; V[k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1; U[1][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1; U[4][col] = -1
        V[p + k0][col] = 1; V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Schoolbook for A[0:2, 2] * B[2, k0:k1+1]
        # A elements: A[0,2]=U[2], A[1,2]=U[5]
        # B elements: B[2,k0]=V[2p+k0], B[2,k1]=V[2p+k1]
        
        # P1 = A02 * B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
