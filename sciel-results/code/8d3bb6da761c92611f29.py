"""Matrix multiplication 2x2 * 2x12 via Strassen (rank 42)."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = 2, 2, 12
    num_groups = p // 2  # 6
    rank = num_groups * 7  # 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 factors
    # U2[i, s] for i in {0,1}, s in {0..6}
    U2 = np.array([
        [ 1,  0,  1,  0,  1, -1,  1],  # i=0
        [ 1,  1,  0,  1,  0,  1, -1],  # i=1
    ])
    
    # V2[j, s] for j in {0,1}, s in {0..6}
    V2 = np.array([
        [ 1,  1, -1,  1,  0,  1,  1],  # j=0
        [ 1,  0,  1, -1,  1,  1,  1],  # j=1
    ])
    
    # W2[lk, s] for lk in {0,1}, s in {0..6}
    # W2[lk, s] is coefficient of C[i, lk] from product s
    # C[0,0] = p1+p4-p5+p7 -> lk=0: p1:1, p2:0, p3:0, p4:1, p5:-1, p6:0, p7:1
    # C[0,1] = p3+p5        -> lk=1: p1:0, p2:0, p3:1, p4:0, p5:1,  p6:0, p7:0
    # C[1,0] = p2+p4        -> lk=0 for row i=1... 
    # Wait, W2 is indexed by output column lk, not by (i,lk). The i index is handled by U2.
    # So W2[lk, s] is the same for all i? No!
    
    # Let me re-derive. In tensor decomposition:
    # T[i, j, k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    # For 2x2x2: T[i,j,k] = A[i,j]*B[j,k] = C[i,k]
    # W[k,i,s] means W is indexed by (k, i) where k is output col, i is input row.
    
    # C[0,0] = p1+p4-p5+p7: contributes to W[0,0,s]
    # C[0,1] = p3+p5:        contributes to W[1,0,s]  
    # C[1,0] = p2+p4:        contributes to W[0,1,s]
    # C[1,1] = p1-p2+p3+p6:  contributes to W[1,1,s]
    
    # So W[k, i, s]:
    # W[0,0]: p1:1, p2:0, p3:0, p4:1, p5:-1, p6:0, p7:1
    # W[1,0]: p1:0, p2:0, p3:1, p4:0, p5:1,  p6:0, p7:0
    # W[0,1]: p1:0, p2:1, p3:0, p4:1, p5:0,  p6:0, p7:0
    # W[1,1]: p1:1, p2:-1,p3:1, p4:0, p5:0,  p6:1, p7:0
    
    W2_4d = np.array([
        [[1,  0,  0, 1, -1,  0,  1],   # W[0,0,s]
         [0,  0,  1, 0,  1,  0,  0]],   # W[1,0,s]
        [[0,  1,  0, 1,  0,  0,  0],   # W[0,1,s]
         [1, -1, 1, 0,  0,  1,  0]]    # W[1,1,s]
    ])
    
    # W2_4d[k, i, s] where k in {0,1}, i in {0,1}, s in {0..6}
    
    for g in range(num_groups):
        for s in range(7):
            r = g * 7 + s
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = U2[i, s]
            for j in range(m):
                for lk in range(2):
                    k_global = 2 * g + lk
                    V[j * p + k_global, r] = V2[j, s]
            for i in range(n):
                for lk in range(2):
                    k_global = 2 * g + lk
                    W[k_global * n + i, r] = W2_4d[lk, i, s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
