import sys
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1.0

    def verify(U, V, W):
        rank = U.shape[1]
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)

    # Helper to add Strassen 2x2 by 2x2 decomposition (Rank 7)
    def add_strassen_2x2_2x2(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x2: j in j_start..j_start+1, k in k_start..k_start+1
        # C is 2x2: k in k_start..k_start+1, i in 0..1
        
        u_a = np.zeros((4, 7))
        u_a[0, 0] = 1; u_a[3, 0] = 1
        u_a[2, 1] = 1; u_a[3, 1] = 1
        u_a[0, 2] = 1
        u_a[3, 3] = 1
        u_a[0, 4] = 1; u_a[1, 4] = 1
        u_a[2, 5] = 1; u_a[0, 5] = -1
        u_a[1, 6] = 1; u_a[3, 6] = -1
        
        v_b = np.zeros((4, 7))
        v_b[0, 0] = 1; v_b[3, 0] = 1
        v_b[0, 1] = 1
        v_b[2, 2] = 1; v_b[3, 2] = -1
        v_b[1, 3] = 1; v_b[0, 3] = -1
        v_b[3, 4] = 1
        v_b[0, 5] = 1; v_b[1, 5] = 1
        v_b[2, 6] = 1; v_b[3, 6] = 1
        
        w_c = np.zeros((4, 7))
        w_c[0, 0] = 1; w_c[1, 0] = 1; w_c[2, 0] = 1; w_c[3, 0] = 1
        w_c[0, 1] = 1; w_c[3, 1] = 1
        w_c[2, 2] = 1; w_c[1, 2] = 1
        w_c[1, 3] = 1; w_c[2, 3] = 1
        w_c[1, 4] = 1; w_c[3, 4] = 1
        w_c[0, 5] = 1; w_c[3, 5] = 1
        w_c[2, 6] = 1; w_c[1, 6] = 1

        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_i = li
                    global_j = j_start + lj
                    U[global_i * m + global_j, col] += u_a[local_idx, r]
                    
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_j = j_start + lj
                    global_k = k_start + lk
                    V[global_j * p + global_k, col] += v_b[local_idx, r]
                    
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_k = k_start + lk
                    global_i = li
                    W[global_k * n + global_i, col] += w_c[local_idx, r]

    # Helper to add 2x2 by 2x1 decomposition (Rank 2)
    def add_2x2_2x1(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x1: j in j_start..j_start+1, k = k_start
        # C is 2x1: k = k_start, i in 0..1
        
        # r=0: A_i0 B_00
        col = col_offset
        for i in range(2):
            U[i * m + j_start, col] += 1.0
        V[j_start * p + k_start, col] += 1.0
        for i in range(2):
            W[k_start * n + i, col] += 1.0
            
        # r=1: A_i1 B_10
        col = col_offset + 1
        for i in range(2):
            U[i * m + (j_start + 1), col] += 1.0
        V[(j_start + 1) * p + k_start, col] += 1.0
        for i in range(2):
            W[k_start * n + i, col] += 1.0

    # Helper to add 2x2 by 2x3 decomposition (Rank 9)
    def add_2x2_2x3(U, V, W, j_start, k_start, col_offset):
        # A is 2x2: i in 0..1, j in j_start..j_start+1
        # B is 2x3: j in j_start..j_start+1, k in k_start..k_start+2
        # C is 2x3: k in k_start..k_start+2, i in 0..1
        
        # Split B into B1 (2x2) and B2 (2x1)
        # B1: k in k_start..k_start+1
        # B2: k = k_start+2
        
        # A * B1 (2x2 * 2x2) -> Rank 7
        add_strassen_2x2_2x2(U, V, W, j_start, k_start, col_offset)
        col_offset += 7
        
        # A * B2 (2x2 * 2x1) -> Rank 2
        add_2x2_2x1(U, V, W, j_start, k_start + 2, col_offset)
        col_offset += 2
        
        return col_offset

    # Helper to add 2x4 by 4x3 decomposition (Rank 18)
    def add_2x4_4x3(U, V, W, j_start, k_start, col_offset):
        # A is 2x4: i in 0..1, j in j_start..j_start+3
        # B is 4x3: j in j_start..j_start+3, k in k_start..k_start+2
        # C is 2x3: k in k_start..k_start+2, i in 0..1
        
        # Split A into A1 (2x2) and A2 (2x2)
        # A1: j in j_start..j_start+1
        # A2: j in j_start+2..j_start+3
        
        # Split B into B1 (2x3) and B2 (2x3)
        # B1: j in j_start..j_start+1, k in k_start..k_start+2
        # B2: j in j_start+2..j_start+3, k in k_start..k_start+2
        
        # A1 * B1 (2x2 * 2x3) -> Rank 9
        col_offset = add_2x2_2x3(U, V, W, j_start, k_start, col_offset)
        
        # A2 * B2 (2x2 * 2x3) -> Rank 9
        col_offset = add_2x2_2x3(U, V, W, j_start + 2, k_start, col_offset)
        
        return col_offset

    # Initialize U, V, W
    # Total Rank = 6 (2x1 * 1x6) + 18 (2x4 * 4x3) + 18 (2x4 * 4x3) = 42
    total_rank = 42
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col_off = 0
    
    # Part 1: A1 (2x1) * B1 (1x6)
    # A1: j=0
    # B1: j=0, k=0..5
    for k in range(6):
        col = col_off + k
        for i in range(2):
            U[i * m + 0, col] += 1.0
        V[0 * p + k, col] += 1.0
        for i in range(2):
            W[k * n + i, col] += 1.0
    col_off += 6
    
    # Part 2: A2 (2x4) * B2 (4x6)
    # A2: j=1..4
    # B2: j=1..4, k=0..5
    
    # Split B2 into B21 (4x3) and B22 (4x3)
    # B21: j=1..4, k=0..2
    # B22: j=1..4, k=3..5
    
    # A2 * B21 (2x4 * 4x3) -> Rank 18
    col_off = add_2x4_4x3(U, V, W, 1, 0, col_off)
    
    # A2 * B22 (2x4 * 4x3) -> Rank 18
    col_off = add_2x4_4x3(U, V, W, 1, 3, col_off)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U0 = np.zeros((nm, rank))
        V0 = np.zeros((mp, rank))
        W0 = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    U0[idx1, col] = 1.0
                    V0[idx2, col] = 1.0
                    W0[idx3, col] = 1.0
                    col += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

# EVOLVE-BLOCK-END
