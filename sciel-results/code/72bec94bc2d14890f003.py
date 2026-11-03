import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part A: j=0, standard terms for each (i,k)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # Part B: j=1,2 using optimized 2x2 block decomposition
    # Process pairs of columns (k0, k1) for the 2x2 subproblem
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # W indices for the 2x2 output block
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # U indices for A rows i=0,1 and columns j=1,2
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        # V indices for B rows j=1,2 and columns k0,k1
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1; V[v_11, col] += -1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1; W[w_01, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1; U[u_11, col] += -1
        V[v_11, col] += 1; V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
