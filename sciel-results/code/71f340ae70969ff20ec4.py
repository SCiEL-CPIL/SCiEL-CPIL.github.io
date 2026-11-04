import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strassen 2x2 factors (for 2x2 * 2x2 -> 2x2)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Treat 2x4 * 4x8 as 2x2 over ring Z[x]/(x^4-1)
    # A[i,j] = sum_{d=0}^{3} A[i,j,d] * x^d  (j = 2*g_j + d, g_j in {0,1})
    # B[j,k] = sum_{e=0}^{3} B[j,k,e] * x^e  (k = 2*g_k + e, g_k in {0..3})
    # C[i,k] = sum_{f=0}^{3} C[i,k,f] * x^f  (k = 2*g_k + f)
    #
    # We do 2x2 Strassen over the ring, where each ring element is a degree-3 poly.
    # Ring multiplication: cyclic convolution mod x^4-1
    # But we only need coefficients 0..3 of the result, and since the true product
    # of two degree-3 polynomials has degree at most 6, the coefficients 0..3 are
    # exactly the non-wrapped terms. The wrapped terms (from x^4, x^5, x^6) don't
    # appear in positions 0..3 of the final answer.
    #
    # So: for each Strassen factor r (7 total), we need 4 "scalar" multiplications
    # (one per coefficient of the ring element). Total rank = 7 * 4 = 28.
    
    rank = 28
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (32, 28)
    # W shape: (p*n, rank) = (16, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(7):
        u_vals = strassen_U[r]  # 4 values: u[0*2+0], u[0*2+1], u[1*2+0], u[1*2+1]
        v_vals = strassen_V[r]  # 4 values: v[0*2+0], v[0*2+1], v[1*2+0], v[1*2+1]
        w_vals = strassen_W[r]  # 4 values: w[0*2+0], w[0*2+1], w[1*2+0], w[1*2+1]
        
        for d in range(4):
            # For coefficient d of the ring element:
            # U[i, j] where j = 2*g_j + d
            #   u_vals[i*2 + d] gives the coefficient
            for i in range(2):
                for g_j in range(2):
                    j_global = 2 * g_j + d
                    u_idx_global = i * m + j_global
                    U[u_idx_global][col] = u_vals[i * 2 + d]
            
            # V[j, k] where j = 2*g_j + d, k = 2*g_k + d
            #   v_vals[g_j*2 + g_k] gives the coefficient
            for g_j in range(2):
                j_global = 2 * g_j + d
                for g_k in range(4):
                    k_global = 2 * g_k + d
                    v_idx_global = j_global * p + k_global
                    V[v_idx_global][col] = v_vals[g_j * 2 + g_k]
            
            # W[k, i] where k = 2*g_k + d
            #   w_vals[g_k*2 + i] gives the coefficient
            for g_k in range(4):
                k_global = 2 * g_k + d
                for i in range(2):
                    w_idx_global = k_global * n + i
                    W[w_idx_global][col] = w_vals[g_k * 2 + i]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
