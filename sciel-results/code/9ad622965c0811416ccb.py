import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms (P1 to P7)
    # u: coeffs for A[00], A[01], A[10], A[11]
    # v: coeffs for B[00], B[01], B[10], B[11]
    # w: coeffs for C[00], C[01], C[10], C[11]
    strassen_terms = [
        # P1
        ([1, 0, -1, 0], [0, 0, 1, -1], [1, 0, 0, 1]),
        # P2
        ([1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1]),
        # P3
        ([0, -1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1]),
        # P4
        ([0, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0]),
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6
        ([1, -1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # P7
        ([0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 0, 0])
    ]
    
    # Loop over blocks
    # a: 0, 1 (split of middle dimension k)
    # b: 0, 1, 2 (split of output dimension j)
    for a in range(2):
        for b in range(3):
            base_col = (a * 3 + b) * 7
            
            for t in range(7):
                col = base_col + t
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                
                # Map local indices to global indices
                # Local 0: 00, 1: 01, 2: 10, 3: 11
                
                for l in range(4):
                    local_row = l // 2
                    local_col = l % 2
                    
                    # U (A): i, k
                    # i = local_row
                    # k = 2*a + local_col
                    i_glob = local_row
                    k_glob = 2 * a + local_col
                    idx_u = i_glob * m + k_glob
                    if u_coeffs[l] != 0:
                        U[idx_u][col] += u_coeffs[l]
                        
                    # V (B): k, j
                    # k = 2*a + local_row
                    # j = 2*b + local_col
                    k_glob_v = 2 * a + local_row
                    j_glob = 2 * b + local_col
                    idx_v = k_glob_v * p + j_glob
                    if v_coeffs[l] != 0:
                        V[idx_v][col] += v_coeffs[l]
                        
                    # W (C): j, i
                    # j = 2*b + local_col
                    # i = local_row
                    j_glob_w = 2 * b + local_col
                    i_glob_w = local_row
                    idx_w = j_glob_w * n + i_glob_w
                    if w_coeffs[l] != 0:
                        W[idx_w][col] += w_coeffs[l]
                        
    return {"U": U, "V": V, "W": W}
