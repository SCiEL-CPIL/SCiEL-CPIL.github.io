"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        if not U or not V or not W:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False
        rank = U_arr.shape[1]
        if rank == 0 or V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Rank-39 decomposition:
    # Split m=3 into j∈{0,1} (2x2 block) and j=2 (2x1 block)
    # For 2x2x9: split into three 2x2x3 blocks, each using Strassen (rank 7) → 21
    # For 2x1x9: schoolbook rank 18
    # Total: 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 27 x 39
    W = [[0] * rank for _ in range(p * n)]  # 18 x 39
    
    col = 0
    # Part 1: j∈{0,1}, 2x2x9 split into three 2x2x3 blocks
    for block in range(3):
        k0 = 3 * block
        k1 = k0 + 1
        k2 = k0 + 2
        
        # Indices for this block
        u_a = 0 * m + 0  # i=0, j=0
        u_b = 0 * m + 1  # i=0, j=1
        u_c = 1 * m + 0  # i=1, j=0
        u_d = 1 * m + 1  # i=1, j=1
        
        v_e = 0 * p + k0  # j=0, k0
        v_f = 0 * p + k1  # j=0, k1
        v_g = 1 * p + k0  # j=1, k0
        v_h = 1 * p + k1  # j=1, k1
        v_i = 0 * p + k2  # j=0, k2
        v_j = 1 * p + k2  # j=1, k2
        
        w_00 = k0 * n + 0  # k0, i=0
        w_01 = k1 * n + 0  # k1, i=0
        w_02 = k2 * n + 0  # k2, i=0
        w_10 = k0 * n + 1  # k0, i=1
        w_11 = k1 * n + 1  # k1, i=1
        w_12 = k2 * n + 1  # k2, i=1
        
        # Strassen 7 multiplications for 2x2x2 block (k0, k1)
        # M1 = (A11+A22)(B11+B22)
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        # M2 = A22*(B21-B11)
        U[u_d][col+1] += 1
        V[v_g][col+1] += 1; V[v_e][col+1] -= 1
        W[w_10][col+1] += 1
        # M3 = (A11-A22)*(B22+B12)
        U[u_a][col+2] += 1; U[u_c][col+2] -= 1
        V[v_h][col+2] += 1; V[v_g][col+2] += 1
        W[w_01][col+2] += 1; W[w_11][col+2] -= 1
        # M4 = A11*(B12-B22)
        U[u_a][col+3] += 1
        V[v_f][col+3] += 1; V[v_h][col+3] -= 1
        W[w_01][col+3] += 1
        # M5 = (A12+A22)*(B11+B12)
        U[u_b][col+4] += 1; U[u_d][col+4] += 1
        V[v_e][col+4] += 1; V[v_f][col+4] += 1
        W[w_00][col+4] += 1; W[w_01][col+4] += 1
        # M6 = (A11+A12)*(B22-B12)
        U[u_a][col+5] += 1; U[u_b][col+5] += 1
        V[v_h][col+5] += 1; V[v_f][col+5] -= 1
        W[w_10][col+5] += 1; W[w_11][col+5] += 1
        # M7 = (A21+A22)*(B11-B21)
        U[u_c][col+6] += 1; U[u_d][col+6] += 1
        V[v_e][col+6] += 1; V[v_g][col+6] -= 1
        W[w_10][col+6] += 1; W[w_11][col+6] -= 1
        
        col += 7
        
        # Handle k2 (third column) with schoolbook (4 terms)
        # C[i,k2] = A[i,0]*B[0,k2] + A[i,1]*B[1,k2]
        U[u_a][col] += 1; V[v_i][col] += 1; W[w_02][col] += 1
        U[u_b][col+1] += 1; V[v_j][col+1] += 1; W[w_02][col+1] += 1
        U[u_c][col+2] += 1; V[v_i][col+2] += 1; W[w_12][col+2] += 1
        U[u_d][col+3] += 1; V[v_j][col+3] += 1; W[w_12][col+3] += 1
        col += 4
    
    # Part 2: j=2, 2x1x9, schoolbook rank 18
    u_ar0 = 0 * m + 2  # i=0, j=2
    u_ar1 = 1 * m + 2  # i=1, j=2
    for k in range(9):
        v_b = 2 * p + k  # j=2, k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    # Verify the decomposition
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col_sb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col_sb] = 1
                V_sb[j * p + k][col_sb] = 1
                W_sb[k * n + i][col_sb] = 1
                col_sb += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
