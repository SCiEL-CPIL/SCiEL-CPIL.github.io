"""Matrix multiplication 2,4,4 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # Rank-28 Strassen-based decomposition
    # Split A (2x4) into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B (4x4) into 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # Apply Strassen 2x2x2 to each of the 4 sub-problems: 7*4 = 28
    
    rank = 28
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def strassen_222(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, col):
        """Add 7 Strassen terms for 2x2x2 multiplication."""
        # T1: (a00+a11)(b00+b11) -> contributes to c00, c11
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # T2: (a10+a11)b00 -> contributes to c10
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # T3: a00(b10-b11) -> contributes to c01
        U[a00, col] += 1
        V[b10, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # T4: a11(b01-b00) -> contributes to c00(-), c10(+)
        U[a11, col] += 1
        V[b01, col] += 1; V[b00, col] -= 1
        W[c00, col] -= 1; W[c10, col] += 1
        col += 1
        
        # T5: (a00+a01)b11 -> contributes to c01(+), c11(-)
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] -= 1
        col += 1
        
        # T6: (a10-a00)(b00+b01) -> contributes to c00(+)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c00, col] += 1
        col += 1
        
        # T7: (a01-a11)(b10+b11) -> contributes to c00(-), c11(+)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] -= 1; W[c11, col] += 1
        col += 1
        
        return col
    
    # Index mappings:
    # U[i*m+j] = U[i*4+j]: A[i,j]
    #   A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    #   A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    # V[j*p+k] = V[j*4+k]: B[j,k]
    #   B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3]
    #   B[1,0]=V[4], B[1,1]=V[5], B[1,2]=V[6], B[1,3]=V[7]
    #   B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
    #   B[3,0]=V[12], B[3,1]=V[13], B[3,2]=V[14], B[3,3]=V[15]
    # W[k*n+i] = W[k*2+i]: C[i,k]
    #   C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    #   C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Sub-problem 1: A1 (cols 0,1) x B1_left (rows 0,1; cols 0,1)
    col = strassen_222(0, 1, 4, 5, 0, 1, 4, 5, 0, 2, 1, 3, col)
    
    # Sub-problem 2: A1 (cols 0,1) x B1_right (rows 0,1; cols 2,3)
    col = strassen_222(0, 1, 4, 5, 2, 3, 6, 7, 4, 6, 5, 7, col)
    
    # Sub-problem 3: A2 (cols 2,3) x B2_left (rows 2,3; cols 0,1)
    col = strassen_222(2, 3, 6, 7, 8, 9, 12, 13, 0, 2, 1, 3, col)
    
    # Sub-problem 4: A2 (cols 2,3) x B2_right (rows 2,3; cols 2,3)
    col = strassen_222(2, 3, 6, 7, 10, 11, 14, 15, 4, 6, 5, 7, col)
    
    assert col == rank, f"Expected {rank} cols, got {col}"
    
    # Verify decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    # Fallback to schoolbook rank 32
                    U_sb = np.zeros((nm, n*m*p))
                    V_sb = np.zeros((mp, n*m*p))
                    W_sb = np.zeros((pn, n*m*p))
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii*m+jj, c] = 1
                                V_sb[jj*p+kk, c] = 1
                                W_sb[kk*n+ii, c] = 1
                                c += 1
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
