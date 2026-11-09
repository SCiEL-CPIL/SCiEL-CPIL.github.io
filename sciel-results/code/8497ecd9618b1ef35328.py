import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a decomposition to the global matrices
    # local_U: list of lists (n*m_local, rank_local)
    # local_V: list of lists (m*p_local, rank_local)
    # local_W: list of lists (p*n_local, rank_local)
    # a_offset, a_stride: mapping for A
    # b_offset, b_stride: mapping for B
    # c_offset, c_stride: mapping for C
    # col_offset: starting column in global U, V, W
    
    def add_decomp(lU, lV, lW, a_map, b_map, c_map, col_off):
        r = len(lU[0]) if lU else 0
        for r_idx in range(r):
            for a_idx, val in enumerate(lU[:, r_idx]):
                if val != 0:
                    global_a = a_map[a_idx]
                    U[global_a][col_off + r_idx] += val
            for b_idx, val in enumerate(lV[:, r_idx]):
                if val != 0:
                    global_b = b_map[b_idx]
                    V[global_b][col_off + r_idx] += val
            for c_idx, val in enumerate(lW[:, r_idx]):
                if val != 0:
                    global_c = c_map[c_idx]
                    W[global_c][col_off + r_idx] += val

    # Generate 2x2x2 Strassen
    def get_strassen():
        # U: 4x7, V: 4x7, W: 4x7
        lU = [[0]*7 for _ in range(4)]
        lV = [[0]*7 for _ in range(4)]
        lW = [[0]*7 for _ in range(4)]
        
        # M1
        lU[0][0] = 1; lU[3][0] = 1
        lV[0][0] = 1; lV[3][0] = 1
        lW[0][0] = 1; lW[3][0] = 1
        
        # M2
        lU[2][1] = 1; lU[3][1] = 1
        lV[0][1] = 1
        lW[2][1] = 1; lW[3][1] = -1
        
        # M3
        lU[0][2] = 1
        lV[1][2] = 1; lV[3][2] = -1
        lW[1][2] = 1; lW[3][2] = 1
        
        # M4
        lU[3][3] = 1
        lV[2][3] = 1; lV[0][3] = -1
        lW[0][3] = 1; lW[2][3] = 1
        
        # M5
        lU[0][4] = 1; lU[1][4] = 1
        lV[3][4] = 1
        lW[0][4] = -1; lW[1][4] = 1
        
        # M6
        lU[2][5] = 1; lU[0][5] = -1
        lV[0][5] = 1; lV[1][5] = 1
        lW[3][5] = 1
        
        # M7
        lU[1][6] = 1; lU[3][6] = -1
        lV[2][6] = 1; lV[3][6] = 1
        lW[0][6] = 1
        
        return lU, lV, lW

    # Generate 2x2x1 Schoolbook
    def get_schoolbook_221():
        lU = [[0]*4 for _ in range(4)]
        lV = [[0]*4 for _ in range(2)] # B is 2x1
        lW = [[0]*4 for _ in range(2)] # C is 2x1
        
        # M1: a00 b00 -> c0
        lU[0][0] = 1
        lV[0][0] = 1
        lW[0][0] = 1
        
        # M2: a01 b10 -> c0
        lU[1][1] = 1
        lV[1][1] = 1
        lW[0][1] = 1
        
        # M3: a10 b00 -> c1
        lU[2][2] = 1
        lV[0][2] = 1
        lW[1][2] = 1
        
        # M4: a11 b10 -> c1
        lU[3][3] = 1
        lV[1][3] = 1
        lW[1][3] = 1
        
        return lU, lV, lW

    # Generate 2x2x3 (2x2x1 + 2x2x2)
    def get_223():
        U1, V1, W1 = get_schoolbook_221()
        U2, V2, W2 = get_strassen()
        
        # Concatenate
        lU = [row[:] for row in U1] + [row[:] for row in U2]
        # V1 is 2x4, V2 is 4x7. Need to align rows.
        # V1 rows: 0,1. V2 rows: 0,1,2,3.
        # We need to create a 4x11 matrix.
        lV = [[0]*11 for _ in range(4)]
        for i in range(2):
            for j in range(4):
                lV[i][j] = V1[i][j]
        for i in range(4):
            for j in range(7):
                lV[i][4+j] = V2[i][j]
                
        lW = [row[:] for row in W1] + [row[:] for row in W2]
        # W1 is 2x4, W2 is 4x7.
        # We need 4x11 matrix.
        # W1 cols 0,1 of C. W2 cols 0,1,2,3 of C (which map to global 2,3,4?).
        # Wait, in local 2x2x3, C is 2x3.
        # C1 (from 221) is col 0.
        # C2 (from 222) is cols 1,2.
        # So W1 should map to local C index 0.
        # W2 should map to local C indices 1,2.
        # But get_strassen returns W for 2x2 output.
        # We need to shift the W indices for the 2x2 part.
        
        # Let's rebuild W properly.
        lW = [[0]*11 for _ in range(3)] # C is 2x3 -> 6 entries? No, 2x3 is 6.
        # My previous W1 was 2x4 (rank 4). Rows 0,1.
        # My previous W2 was 4x7 (rank 7). Rows 0,1,2,3.
        
        # Local C indices: 0, 1, 2.
        # C_0 comes from 221.
        # C_1, C_2 come from 222.
        
        # Fill W
        for r in range(4):
            lW[0][r] = W1[0][r]
            lW[1][r] = W1[1][r]
            
        for r in range(7):
            global_r = 4 + r
            lW[1][global_r] += W2[0][r]
            lW[2][global_r] += W2[1][r]
            lW[3][global_r] += W2[2][r] # Wait, 2x2x3 C is 2x3. Indices 0,1,2.
            # W2 is for 2x2 output. Indices 0,1.
            # So W2[0] -> C_1, W2[1] -> C_2.
            # W2[2], W2[3] are unused?
            # No, 2x2 output has 4 entries.
            # 2x2x3 output has 6 entries? No, 2 rows, 3 cols. 6 entries.
            # But the 2x2x2 part only produces 4 entries.
            # The 2x2x1 part produces 2 entries.
            # Total 6.
            # So W should be 6x11.
            
            # Correction:
            # C is 2x3. Flattened 0..5.
            # C_0 (0,0) from 221.
            # C_1 (0,1) from 222 (0,0).
            # C_2 (0,2) from 222 (0,1).
            # C_3 (1,0) from 221.
            # C_4 (1,1) from 222 (1,0).
            # C_5 (1,2) from 222 (1,1).
            
            pass # Re-implement in code

    return {"U": U, "V": V, "W": W}
