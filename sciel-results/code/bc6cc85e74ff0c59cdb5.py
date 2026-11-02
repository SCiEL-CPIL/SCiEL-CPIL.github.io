import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Rank 30 Decomposition
    total_rank = 30
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    def add_strassen_2x2x2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Strassen patterns for 2x2x2
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2
                    j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx // 2
                    k_l = idx % 2
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx // 2
                    k_l = idx % 2
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            
            col_off += 1
            
        return col_off

    def add_rank1_2x2x1(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Rank 1 for 2x2x1
        # C_ik = A_i0 * B_0k
        # U_i = 1, V_0 = 1, W_k = 1
        for i in range(2):
            U[(a_i_off + i) * m + a_j_off, col_off] = 1
        V[(b_j_off) * p + b_k_off, col_off] = 1
        for k in range(2):
            W[(c_k_off + k) * n + c_i_off, col_off] = 1
            
        return col_off + 1

    def add_2x2x5_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Block: A (2x2) * B (2x5)
        # Split B into B1 (2x2) and B2 (2x3)
        # B1 is cols 0-1 of local B
        # B2 is cols 2-4 of local B
        
        # Part 1: A * B1 (2x2x2)
        col_off = add_strassen_2x2x2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off)
        
        # Part 2: A * B2 (2x2x3)
        # Split B2 into B21 (2x2) and B22 (2x1)
        # B21 is cols 2-3 of local B
        # B22 is col 4 of local B
        
        # Part 2a: A * B21 (2x2x2)
        col_off = add_strassen_2x2x2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off + 2, c_i_off, c_k_off + 2, col_off)
        
        # Part 2b: A * B22 (2x2x1)
        col_off = add_rank1_2x2x1(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off + 4, c_i_off, c_k_off + 4, col_off)
        
        return col_off

    col = 0
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    col = add_2x2x5_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    col = add_2x2x5_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    
    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"
    
    # Verify
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check non-zero terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zero terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
