"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # all 5
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2 Strassen (rank 7).
        Row index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in row-major order for the 2x2 matrix.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1  # C00
        W[3][0] += 1  # C11
        
        # M2 = (A10 + A11) B00
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[1][1] += 1  # C01
        
        # M3 = A00 (B01 - B11)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] += -1
        W[2][2] += 1  # C10
        
        # M4 = A11 (B10 - B00)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] += -1
        W[0][3] += 1  # C00
        W[2][3] += 1  # C10
        
        # M5 = (A00 + A01) B11
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[3][4] += 1  # C11
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[1][5] += 1
        W[0][5] += -1  # C00
        W[1][5] += 1   # C01
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][6] += 1; U[3][6] += -1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1   # C00
        W[3][6] += 1   # C11
        
        return U, V, W
    
    def strassen_4x4():
        """Returns U, V, W for 4x4 Strassen (rank 49) via tensor product of two 2x2 Strassen.
        Row index r in 0..15 corresponds to flat index in 4x4: r = row*4 + col where row,col in 0..3.
        Column index t in 0..48: t = t_meta*7 + t_leaf
        """
        U2, V2, W2 = get_strassen_2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for t_meta in range(7):
            for t_leaf in range(7):
                t = t_meta * 7 + t_leaf
                for r_meta in range(2):
                    for c_meta in range(2):
                        for r_leaf in range(2):
                            for c_leaf in range(2):
                                r = (r_meta * 2 + r_leaf) * 4 + (c_meta * 2 + c_leaf)
                                u_val = U2[r_meta * 2 + r_leaf][t_meta] * U2[c_meta * 2 + c_leaf][t_leaf]
                                if u_val != 0:
                                    U[r][t] += u_val
                                v_val = V2[r_meta * 2 + r_leaf][t_meta] * V2[c_meta * 2 + c_leaf][t_leaf]
                                if v_val != 0:
                                    V[r][t] += v_val
                                w_val = W2[r_meta * 2 + r_leaf][t_meta] * W2[c_meta * 2 + c_leaf][t_leaf]
                                if w_val != 0:
                                    W[r][t] += w_val
        return U, V, W
    
    try:
        # Build the full 5x5 decomposition
        # Partition: indices 0-3 (4x4 block) and index 4 (the 5th)
        # C_44 = A_44 * B_44: rank 49 (Strassen)
        # C_41 = A_44 * B_41 + A_41 * B_11: rank 16 + 4 = 20
        # C_14 = A_14 * B_44 + A_11 * B_14: rank 16 + 4 = 20
        # C_11 = A_14 * B_41 + A_11 * B_11: rank 4 + 1 = 5
        # Total: 49 + 20 + 20 + 5 = 94
        
        total_rank = 94
        U = [[0] * total_rank for _ in range(25)]
        V = [[0] * total_rank for _ in range(25)]
        W = [[0] * total_rank for _ in range(25)]
        
        col = 0
        
        # Part 1: C_44 = A_44 * B_44 using Strassen 4x4 (rank 49)
        U4, V4, W4 = strassen_4x4()
        for r in range(16):
            row = r // 4
            col_idx = r % 4
            full_row = row * 5 + col_idx
            for t in range(49):
                U[full_row][col + t] = U4[r][t]
                V[full_row][col + t] = V4[r][t]
                W[full_row][col + t] = W4[r][t]
        col += 49
        
        if time.time() - start_time > timeout:
            return schoolbook()
        
        # Part 2: C_41 = A_44 * B_41 + A_41 * B_11 (20 terms)
        for i in range(4):
            for j in range(4):
                # A[i,j] * B[j,4] -> C[i,4]
                U[i * 5 + j][col] = 1
                V[j * 5 + 4][col] = 1
                W[4 * 5 + i][col] = 1
                col += 1
        
        for i in range(4):
            # A[i,4] * B[4,4] -> C[i,4]
            U[i * 5 + 4][col] = 1
            V[4 * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
        
        if time.time() - start_time > timeout:
            return schoolbook()
        
        # Part 3: C_14 = A_14 * B_44 + A_11 * B_14 (20 terms)
        for j in range(4):
            for k in range(4):
                # A[4,j] * B[j,k] -> C[4,k]
                U[4 * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + 4][col] = 1
                col += 1
        
        for k in range(4):
            # A[4,4] * B[4,k] -> C[4,k]
            U[4 * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
        
        if time.time() - start_time > timeout:
            return schoolbook()
        
        # Part 4: C_11 = A_14 * B_41 + A_11 * B_11 (5 terms)
        for j in range(4):
            # A[4,j] * B[j,4] -> C[4,4]
            U[4 * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + 4][col] = 1
            col += 1
        
        # A[4,4] * B[4,4] -> C[4,4]
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
        if col != total_rank:
            return schoolbook()
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        return schoolbook()
# EVOLVE-BLOCK-END
