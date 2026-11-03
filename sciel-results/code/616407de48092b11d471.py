import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to build Strassen 2x2x2 factors
    def strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1 = A00(B00+B11)
        u[0][0] = 1; v[0][0] = 1; v[3][0] = 1
        # M2 = (A10+A11)B00
        u[2][1] = 1; u[3][1] = 1; v[0][1] = 1
        # M3 = A00(B01-B11)
        u[0][2] = 1; v[1][2] = 1; v[3][2] = -1
        # M4 = A10 B11
        u[2][3] = 1; v[3][3] = 1
        # M5 = (A00+A01)B10
        u[0][4] = 1; u[1][4] = 1; v[2][4] = 1
        # M6 = (A10-A00)(B00+B01)
        u[2][5] = 1; u[0][5] = -1; v[0][5] = 1; v[1][5] = 1
        # M7 = (A01-A11)(B10+B11)
        u[1][6] = 1; u[3][6] = -1; v[2][6] = 1; v[3][6] = 1
        
        # W: C00 = M1 - M4 + M6 - M7
        w[0][0] = 1; w[0][3] = -1; w[0][5] = 1; w[0][6] = -1
        # C01 = M2 + M3
        w[1][1] = 1; w[1][2] = 1
        # C10 = M4 + M5
        w[2][3] = 1; w[2][4] = 1
        # C11 = M1 - M2 + M5 + M7
        w[3][0] = 1; w[3][1] = -1; w[3][4] = 1; w[3][6] = 1
        
        return u, v, w

    # Helper to build Schoolbook 2x2x1 factors
    def schoolbook_2x2x1():
        # U: 4x4, V: 4x4, W: 2x4
        u = [[0]*4 for _ in range(4)]
        v = [[0]*4 for _ in range(4)]
        w = [[0]*4 for _ in range(2)]
        
        # C00 = A00B00 + A01B10
        u[0][0] = 1; v[0][0] = 1; w[0][0] = 1
        u[1][1] = 1; v[2][1] = 1; w[0][1] = 1
        # C10 = A10B00 + A11B10
        u[2][2] = 1; v[0][2] = 1; w[1][2] = 1
        u[3][3] = 1; v[2][3] = 1; w[1][3] = 1
        
        return u, v, w

    # Build 2x2x3 factors (Rank 11)
    u_222, v_222, w_222 = strassen_2x2x2()
    u_221, v_221, w_221 = schoolbook_2x2x1()
    
    u_223 = [u_222[i] + u_221[i] for i in range(4)]
    v_223 = [v_222[i] + v_221[i] for i in range(4)]
    # W is 6x11
    w_223 = [[0]*11 for _ in range(6)]
    for i in range(4):
        for c in range(7):
            w_223[i][c] = w_222[i][c]
    for i in range(2):
        for c in range(4):
            w_223[4+i][7+c] = w_221[i][c]

    # Build 2x4x3 factors (Rank 22)
    # T1: j in 0,1. T2: j in 2,3.
    u_243 = [[0]*22 for _ in range(8)]
    v_243 = [[0]*22 for _ in range(12)]
    w_243 = [[0]*22 for _ in range(6)]
    
    # T1 (cols 0-10)
    for i in range(4):
        u_243[i][0:11] = u_223[i]
        v_243[i][0:11] = v_223[i]
    for i in range(6):
        w_243[i][0:11] = w_223[i]
        
    # T2 (cols 11-21)
    for i in range(4):
        u_243[4+i][11:22] = u_223[i]
        v_243[6+i][11:22] = v_223[i]
    for i in range(6):
        w_243[i][11:22] = w_223[i]

    # Build 4x4x6 factors (Rank 88)
    # 4 sub-problems:
    # 11: i in 0,1, k in 0,1,2
    # 12: i in 0,1, k in 3,4,5
    # 21: i in 2,3, k in 0,1,2
    # 22: i in 2,3, k in 3,4,5
    
    col_offset = 0
    for i_block in [0, 2]:
        for k_block in [0, 3]:
            # U: rows i_block..i_block+1, j 0..3
            for i in range(4):
                U[i_block*2 + i//2][col_offset:col_offset+22] = u_243[i][0:22] # Wait, u_243 is 8x22.
                # Correct mapping:
                # u_243 rows 0-3 correspond to i=0,1 and j=0,1
                # u_243 rows 4-7 correspond to i=0,1 and j=2,3
                # We need to place this into U which has rows for i=0..3, j=0..3
                # For i_block=0 (i=0,1):
                #   j=0,1 -> u_243 rows 0-3
                #   j=2,3 -> u_243 rows 4-7
                # For i_block=2 (i=2,3):
                #   j=0,1 -> u_243 rows 0-3 (but shifted in i)
                #   j=2,3 -> u_243 rows 4-7
            
            # Let's do it explicitly
            for j in range(4):
                local_i = 0 if i_block == 0 else 1 # No, local i is 0,1
                # u_243 index: row = local_i * 4 + j
                # U index: row = (i_block + local_i) * 4 + j
                for local_i in range(2):
                    row_u = (i_block + local_i) * 4 + j
                    src_row = local_i * 4 + j
                    U[row_u][col_offset:col_offset+22] = u_243[src_row]
            
            # V: rows j 0..3, k_block..k_block+2
            for j in range(4):
                for local_k in range(3):
                    row_v = j * 6 + (k_block + local_k)
                    src_row = j * 3 + local_k
                    V[row_v][col_offset:col_offset+22] = v_243[src_row]
            
            # W: rows k_block..k_block+2, i_block..i_block+1
            for local_k in range(3):
                for local_i in range(2):
                    row_w = (k_block + local_k) * 4 + (i_block + local_i)
                    src_row = local_k * 2 + local_i
                    W[row_w][col_offset:col_offset+22] = w_243[src_row]
            
            col_offset += 22

    return {"U": U, "V": V, "W": W}
