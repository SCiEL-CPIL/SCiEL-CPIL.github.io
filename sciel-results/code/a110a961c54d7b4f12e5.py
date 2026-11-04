import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (upper bound) exact tensor decomposition for 
    multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    1. Decompose M_4,4,4 using recursive Strassen (Rank 49).
    2. Use two such blocks for the first 8 columns of B.
    3. Decompose M_4,4,1 using Identity (Rank 16) for the 9th column.
    Total Rank = 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Helper to generate M_2,2,2 Strassen decomposition
    def get_m2_strassen():
        # Returns U, V, W as lists of lists (4x7)
        # Indices for U/V/W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # s=0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # s=1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # s=2: M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[0][2] = -1; W[1][2] = 1
        
        # s=3: M4 = A11(B10-B00)
        U[3][3] = 1
        V[0][3] = -1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1
        
        # s=4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # s=5: M6 = (A10-A00)(B00+B01)
        U[0][5] = -1; U[2][5] = 1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # s=6: M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    # Helper to generate M_4,4,4 decomposition using M_2 Strassen
    def get_m4_strassen():
        U2, V2, W2 = get_m2_strassen()
        
        # Coefficients for A_eff and B_eff blocks
        # r in 0..6, block (I,J) in 0..1
        A_coeff = [
            [1,0,0,1], # M1
            [0,0,1,1], # M2
            [1,0,0,0], # M3
            [0,0,0,1], # M4
            [1,1,0,0], # M5
            [-1,0,1,0],# M6
            [0,1,0,-1] # M7
        ]
        B_coeff = [
            [1,0,0,1], # M1
            [1,0,0,0], # M2
            [0,1,0,-1],# M3
            [-1,0,1,0],# M4
            [0,0,0,1], # M5
            [1,1,0,0], # M6
            [0,0,1,1]  # M7
        ]
        # C contribution coefficients W_add[r, I, J]
        # C00: M1+M4-M5+M7, C01: M3+M5, C10: M2+M4, C11: M1-M2+M3+M6
        W_add = [
            [1, 0, 0, 1], # r=0: C00=1, C11=1
            [0, 0, 1, -1],# r=1: C10=1, C11=-1
            [-1, 1, 0, 1],# r=2: C00=-1, C01=1, C11=1
            [1, 0, 1, 0], # r=3: C00=1, C10=1
            [-1, 1, 0, 0],# r=4: C00=-1, C01=1
            [0, 0, 0, 1], # r=5: C11=1
            [1, 0, 0, 1]  # r=6: C00=1, C11=1
        ]
        
        rank = 49
        U4 = [[0]*rank for _ in range(16)]
        V4 = [[0]*rank for _ in range(16)]
        W4 = [[0]*rank for _ in range(16)]
        
        col = 0
        for r in range(7):
            for s in range(7):
                # Fill U4
                for I in range(2):
                    for J in range(2):
                        cA = A_coeff[r][I*2+J]
                        if cA == 0: continue
                        for i_loc in range(2):
                            for j_loc in range(2):
                                cU2 = U2[i_loc*2+j_loc][s]
                                if cU2 == 0: continue
                                global_i = I*2 + i_loc
                                global_j = J*2 + j_loc
                                U4[global_i*4+global_j][col] += cA * cU2
                
                # Fill V4
                for I in range(2):
                    for J in range(2):
                        cB = B_coeff[r][I*2+J]
                        if cB == 0: continue
                        for j_loc in range(2):
                            for k_loc in range(2):
                                cV2 = V2[j_loc*2+k_loc][s]
                                if cV2 == 0: continue
                                global_j = I*2 + j_loc
                                global_k = J*2 + k_loc
                                V4[global_j*4+global_k][col] += cB * cV2
                
                # Fill W4
                for I in range(2):
                    for J in range(2):
                        cW_add = W_add[r][I*2+J]
                        if cW_add == 0: continue
                        for k_loc in range(2):
                            for i_loc in range(2):
                                cW2 = W2[k_loc*2+i_loc][s]
                                if cW2 == 0: continue
                                global_k = J*2 + k_loc
                                global_i = I*2 + i_loc
                                # W index: k*n + i = 4*global_k + global_i
                                W4[4*global_k+global_i][col] += cW_add * cW2
                
                col += 1
                
        return U4, V4, W4

    # Helper to generate M_4,4,1 decomposition (Rank 16)
    def get_m4_1():
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for i in range(4):
            for j in range(4):
                col = i*4+j
                U[i*4+j][col] = 1
                # V index: j*p + k. p=1, k=0. So V[4j][col] = 1
                V[j*1+0][col] = 1
                # W index: k*n + i. k=0. So W[i][col] = 1
                W[0*4+i][col] = 1
                
        return U, V, W

    # Build final M_4,4,9
    U4, V4, W4 = get_m4_strassen()
    U4b, V4b, W4b = get_m4_strassen()
    U1, V1, W1 = get_m4_1()
    
    rank = 49 + 49 + 16
    U_final = [[0]*rank for _ in range(16)]
    V_final = [[0]*rank for _ in range(36)] # m*p = 4*9 = 36
    W_final = [[0]*rank for _ in range(36)] # p*n = 9*4 = 36
    
    # Part 1: B cols 0-3
    for i in range(16):
        for r in range(49):
            U_final[i][r] = U4[i][r]
            V_final[i][r] = V4[i][r]
            W_final[i][r] = W4[i][r]
            
    # Part 2: B cols 4-7
    # V index shift: B_{j, k+4}. Index j*9 + (k+4).
    # W index shift: C_{i, k+4}. Index (k+4)*4 + i.
    for i in range(16):
        for r in range(49):
            col = 49 + r
            U_final[i][col] = U4b[i][r]
            # Map V indices
            for old_idx in range(16):
                if V4b[old_idx][r] != 0:
                    # old_idx = j*4 + k
                    j = old_idx // 4
                    k = old_idx % 4
                    new_idx = j*9 + (k+4)
                    V_final[new_idx][col] += V4b[old_idx][r]
            # Map W indices
            for old_idx in range(16):
                if W4b[old_idx][r] != 0:
                    # old_idx = k*4 + i
                    k = old_idx // 4
                    i_loc = old_idx % 4
                    new_idx = (k+4)*4 + i_loc
                    W_final[new_idx][col] += W4b[old_idx][r]
                    
    # Part 3: B col 8
    # V index: B_{j, 8}. Index j*9 + 8.
    # W index: C_{i, 8}. Index 8*4 + i = 32 + i.
    for i in range(16):
        for r in range(16):
            col = 98 + r
            U_final[i][col] = U1[i][r]
            for old_idx in range(16):
                if V1[old_idx][r] != 0:
                    j = old_idx // 1 # p=1, so old_idx = j
                    new_idx = j*9 + 8
                    V_final[new_idx][col] += V1[old_idx][r]
            for old_idx in range(16):
                if W1[old_idx][r] != 0:
                    i_loc = old_idx % 4
                    new_idx = 32 + i_loc
                    W_final[new_idx][col] += W1[old_idx][r]
                    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
