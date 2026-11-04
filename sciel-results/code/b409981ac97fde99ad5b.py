import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_total = 99
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_block(rank_offset, i0, j0, k0, n_b, m_b, p_b, u_mat, v_mat, w_mat):
        # u_mat: n_b * m_b * rank_b
        # v_mat: m_b * p_b * rank_b
        # w_mat: p_b * n_b * rank_b
        # This helper is too generic, I'll stick to specific helpers.
        pass

    def add_strassen(rank_offset, i0, j0, k0):
        # 2x2x2 Strassen
        # U: 2x2, V: 2x2, W: 2x2
        # Rank 7
        u_loc = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        v_loc = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 1, 0, 0, 1, 0],  # B10
            [0, 0, 0, 1, 0, 0, 1],  # B01
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        w_loc = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [0, 0, 1, 0, 1, 0, 0],  # C01
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        # 2x2x1
        # Rank 4
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        u_loc = [
            [1, 0, 0, 0], # A00
            [0, 1, 0, 0], # A01
            [0, 0, 1, 0], # A10
            [0, 0, 0, 1]  # A11
        ]
        v_loc = [
            [1, 0, 1, 0], # B00
            [0, 1, 0, 1]  # B10
        ]
        w_loc = [
            [1, 1, 0, 0], # C0
            [0, 0, 1, 1]  # C1
        ]
        
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(1):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*1 + k_l][r]
        
        for k_l in range(1):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m122(rank_offset, i0, j0, k0):
        # 1x2x2
        # Rank 4
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        u_loc = [
            [1, 0, 0, 0], # A0
            [0, 1, 0, 1]  # A1
        ]
        v_loc = [
            [1, 0, 1, 0], # B00
            [0, 1, 0, 1]  # B10
        ]
        # Wait, B is 2x2.
        # B00, B10 for C0.
        # B01, B11 for C1.
        # v_loc[0] is B00, v_loc[1] is B10?
        # No, V index is j*p+k.
        # B00: 0. B01: 1. B10: 2. B11: 3.
        # C0 = A0 B00 + A1 B10.
        # C1 = A0 B01 + A1 B11.
        
        # r=0: A0 B00. u[0][0]=1. v[0][0]=1. w[0][0]=1.
        # r=1: A1 B10. u[1][1]=1. v[2][1]=1. w[0][1]=1.
        # r=2: A0 B01. u[0][2]=1. v[1][2]=1. w[1][2]=1.
        # r=3: A1 B11. u[1][3]=1. v[3][3]=1. w[1][3]=1.
        
        u_loc = [
            [1, 0, 1, 0], # A0
            [0, 1, 0, 1]  # A1
        ]
        v_loc = [
            [1, 0, 0, 0], # B00
            [0, 0, 1, 0], # B01
            [0, 1, 0, 0], # B10
            [0, 0, 0, 1]  # B11
        ]
        w_loc = [
            [1, 1, 0, 0], # C0
            [0, 0, 1, 1]  # C1
        ]
        
        for i_l in range(1):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(1):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*1 + i_l][r]

    def add_m212(rank_offset, i0, j0, k0):
        # 2x1x2
        # Rank 4
        # C0 = A0 B0 + A1 B0
        # C1 = A0 B1 + A1 B1
        # A is 2x1. B is 1x2.
        # A0, A1. B0, B1.
        # C0 = A0 B0 + A1 B0? No.
        # A = [A0; A1]. B = [B0 B1].
        # C = [A0 B0 + A1 B0? No.
        # C = A * B.
        # C0 = A0 B0 + A1 B0? No.
        # C is 2x2.
        # C00 = A0 B0. C01 = A0 B1.
        # C10 = A1 B0. C11 = A1 B1.
        
        u_loc = [
            [1, 0, 0, 0], # A0
            [0, 1, 0, 1]  # A1
        ]
        v_loc = [
            [1, 0, 1, 0]  # B0 (row 0, col 0 and 1? No, B is 1x2. B0 is col 0, B1 is col 1)
            # B index: j*p+k. p=2.
            # B00: 0. B01: 1.
            # v_loc[0] is B00. v_loc[1] is B01.
        ]
        # r=0: A0 B00. u[0][0]=1. v[0][0]=1. w[0][0]=1.
        # r=1: A1 B00. u[1][1]=1. v[0][1]=1. w[1][1]=1.
        # r=2: A0 B01. u[0][2]=1. v[1][2]=1. w[2][2]=1.
        # r=3: A1 B01. u[1][3]=1. v[1][3]=1. w[3][3]=1.
        
        u_loc = [
            [1, 0, 1, 0], # A0
            [0, 1, 0, 1]  # A1
        ]
        v_loc = [
            [1, 1, 0, 0], # B00
            [0, 0, 1, 1]  # B01
        ]
        w_loc = [
            [1, 0, 0, 0], # C00
            [0, 1, 0, 0], # C10
            [0, 0, 1, 0], # C01
            [0, 0, 0, 1]  # C11
        ]
        
        for i_l in range(2):
            for j_l in range(1):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*1 + j_l][r]
        
        for j_l in range(1):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    # Build the 99 rank decomposition
    # A: 3x5, B: 5x7, C: 3x7
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_strassen(7, 0, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[4:6, 0:2]
    add_strassen(14, 0, 0, 4)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6:7] -> C[6:7, 0:2]
    add_m221(21, 0, 0, 6)
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:7] -> C[0:7, 2:3]
    # Split into 3x M122 and 1x M121?
    # M127 is rank 14.
    # M122 (0:2) rank 4. M122 (2:4) rank 4. M122 (4:6) rank 4. M121 (6:7) rank 2.
    # Total 14.
    add_m122(25, 2, 0, 0)
    add_m122(29, 2, 0, 2)
    add_m122(33, 2, 0, 4)
    # M121: 1x2x1. Rank 2.
    # A[2,0:2] * B[0:2, 6].
    # C[6,2] = A20 B06 + A21 B16.
    # u: A20, A21. v: B06, B16. w: C62.
    # r=0: A20 B06. u[0][0]=1. v[0][0]=1. w[0][0]=1.
    # r=1: A21 B16. u[1][1]=1. v[1][1]=1. w[0][1]=1.
    u_loc = [[1, 0], [0, 1]]
    v_loc = [[1, 0], [0, 1]]
    w_loc = [[1, 1]]
    
    for i_l in range(1):
        for j_l in range(2):
            u_g_idx = (2 + i_l) * m + (0 + j_l)
            for r in range(2):
                U[u_g_idx][37 + r] += u_loc[i_l*2 + j_l][r]
    for j_l in range(2):
        for k_l in range(1):
            v_g_idx = (0 + j_l) * p + (6 + k_l)
            for r in range(2):
                V[v_g_idx][37 + r] += v_loc[j_l*1 + k_l][r]
    for k_l in range(1):
        for i_l in range(1):
            w_g_idx = (6 + k_l) * n + (2 + i_l)
            for r in range(2):
                W[w_g_idx][37 + r] += w_loc[k_l*1 + i_l][r]
                
    # Block 6: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(39, 0, 2, 0)
    
    # Block 7: A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]
    add_strassen(46, 0, 2, 2)
    
    # Block 8: A[0:2, 2:4] * B[2:4, 4:6] -> C[4:6, 0:2]
    add_strassen(53, 0, 2, 4)
    
    # Block 9: A[0:2, 2:4] * B[2:4, 6:7] -> C[6:7, 0:2]
    add_m221(60, 0, 2, 6)
    
    # Block 10: A[2:3, 2:4] * B[2:4, 0:7] -> C[0:7, 2:3]
    add_m122(64, 2, 2, 0)
    add_m122(68, 2, 2, 2)
    add_m122(72, 2, 2, 4)
    # M121 (6:7)
    for i_l in range(1):
        for j_l in range(2):
            u_g_idx = (2 + i_l) * m + (2 + j_l)
            for r in range(2):
                U[u_g_idx][76 + r] += u_loc[i_l*2 + j_l][r]
    for j_l in range(2):
        for k_l in range(1):
            v_g_idx = (2 + j_l) * p + (6 + k_l)
            for r in range(2):
                V[v_g_idx][76 + r] += v_loc[j_l*1 + k_l][r]
    for k_l in range(1):
        for i_l in range(1):
            w_g_idx = (6 + k_l) * n + (2 + i_l)
            for r in range(2):
                W[w_g_idx][76 + r] += w_loc[k_l*1 + i_l][r]

    # Block 11: A[0:3, 4:5] * B[4:5, 0:7] -> C[0:7, 0:3]
    # M317. Rank 21.
    # A[0:3, 4] * B[4, 0:7].
    # C[0:7, 0:3].
    # Each C[i,k] = A[i,4] B[4,k].
    # 21 terms.
    for i_l in range(3):
        for k_l in range(7):
            r = 78 + i_l * 7 + k_l
            u_g_idx = (i_l) * m + (4)
            U[u_g_idx][r] += 1
            v_g_idx = (4) * p + (k_l)
            V[v_g_idx][r] += 1
            w_g_idx = (k_l) * n + (i_l)
            W[w_g_idx][r] += 1

    return {"U": U, "V": V, "W": W}
