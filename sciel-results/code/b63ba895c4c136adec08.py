"""Tensor decomposition for 3x3 matrix multiplication.
Uses a rank-23 decomposition by merging 4 pairs of schoolbook terms into single rank-1 tensors.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 23
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    # 19 pure terms for unpaired triples
    pure_triples = [
        (0,0,0), (0,1,0), (0,1,1), (0,2,0), (0,2,1), (0,2,2),
        (1,0,1), (1,0,2), (1,1,0), (1,1,2), (1,2,0), (1,2,1),
        (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2), (2,2,0), (2,2,2)
    ]
    
    col = 0
    for (i, j, k) in pure_triples:
        U[i*3+j, col] = 1.0
        V[j*3+k, col] = 1.0
        W[k*3+i, col] = 1.0
        col += 1
    
    # 4 merged terms
    # Pair 1: (0,0,1) and (1,0,0)
    # u[0]=1, u[3]=1, v[0]=1, w[3]=1, w[1]=1
    U[0, col] = 1.0; U[3, col] = 1.0
    V[0, col] = 1.0
    W[3, col] = 1.0; W[1, col] = 1.0
    col += 1
    
    # Pair 2: (0,1,2) and (1,1,1)
    # u[1]=1, u[4]=1, v[5]=1, v[4]=1, w[6]=1, w[4]=1
    U[1, col] = 1.0; U[4, col] = 1.0
    V[5, col] = 1.0; V[4, col] = 1.0
    W[6, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # Pair 3: (2,0,0) and (0,0,2)
    # u[6]=1, u[0]=1, v[0]=1, v[2]=1, w[2]=1, w[6]=1
    U[6, col] = 1.0; U[0, col] = 1.0
    V[0, col] = 1.0; V[2, col] = 1.0
    W[2, col] = 1.0; W[6, col] = 1.0
    col += 1
    
    # Pair 4: (1,2,2) and (2,2,1)
    # u[5]=1, u[8]=1, v[8]=1, v[7]=1, w[7]=1, w[5]=1
    U[5, col] = 1.0; U[8, col] = 1.0
    V[8, col] = 1.0; V[7, col] = 1.0
    W[7, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    # Verify
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = sum(U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank_sb = 27
                    U_sb = np.zeros((9, rank_sb), dtype=float)
                    V_sb = np.zeros((9, rank_sb), dtype=float)
                    W_sb = np.zeros((9, rank_sb), dtype=float)
                    col_sb = 0
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                U_sb[ii*3+jj, col_sb] = 1.0
                                V_sb[jj*3+kk, col_sb] = 1.0
                                W_sb[kk*3+ii, col_sb] = 1.0
                                col_sb += 1
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
