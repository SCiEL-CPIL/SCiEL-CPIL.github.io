"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 rank 7 decomposition
    def add_strassen(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u11, u12, u21, u22]
        # v_idx: [v11, v12, v21, v22]
        # w_idx: [w11, w12, w21, w22]
        
        # P1 = (u11+u22)(v11+v22)
        U[u_idx[0]][col] = 1; V[v_idx[0]][col] = 1
        U[u_idx[3]][col] = 1; V[v_idx[3]][col] = 1
        W[w_idx[0]][col] = 1; W[w_idx[1]][col] = 1; W[w_idx[2]][col] = 1; W[w_idx[3]][col] = 1
        col += 1
        
        # P2 = (u21+u22)v11
        U[u_idx[2]][col] = 1; V[v_idx[0]][col] = 1
        U[u_idx[3]][col] = 1
        W[w_idx[2]][col] = 1; W[w_idx[3]][col] = -1
        col += 1
        
        # P3 = u11(v12-v22)
        U[u_idx[0]][col] = 1; V[v_idx[1]][col] = 1
        V[v_idx[3]][col] = -1
        W[w_idx[1]][col] = 1; W[w_idx[3]][col] = 1
        col += 1
        
        # P4 = u22(v21-v11)
        U[u_idx[3]][col] = 1; V[v_idx[2]][col] = 1
        V[v_idx[0]][col] = -1
        W[w_idx[0]][col] = 1; W[w_idx[2]][col] = 1
        col += 1
        
        # P5 = (u11+u12)v22
        U[u_idx[0]][col] = 1; V[v_idx[3]][col] = 1
        U[u_idx[1]][col] = 1
        W[w_idx[0]][col] = -1; W[w_idx[1]][col] = 1
        col += 1
        
        # P6 = (u21-u11)(v11+v12)
        U[u_idx[2]][col] = 1; V[v_idx[0]][col] = 1
        U[u_idx[0]][col] = -1; V[v_idx[1]][col] = 1
        W[w_idx[3]][col] = 1
        col += 1
        
        # P7 = (u12-u22)(v21+v22)
        U[u_idx[1]][col] = 1; V[v_idx[2]][col] = 1
        U[u_idx[3]][col] = -1; V[v_idx[3]][col] = 1
        W[w_idx[0]][col] = 1; W[w_idx[1]][col] = -1
        col += 1

    # Helper to add Schoolbook 1x2x2 rank 4 decomposition
    def add_schoolbook_1x2x2(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u0, u1] (1x2)
        # v_idx: [v00, v01, v10, v11] (2x2)
        # w_idx: [w0, w1] (1x2)
        
        # C0 = u0*v00 + u1*v10
        # C1 = u0*v01 + u1*v11
        
        # P1 = u0*v00 -> w0
        U[u_idx[0]][col] = 1; V[v_idx[0]][col] = 1
        W[w_idx[0]][col] = 1
        col += 1
        
        # P2 = u1*v10 -> w0
        U[u_idx[1]][col] = 1; V[v_idx[2]][col] = 1
        W[w_idx[0]][col] = 1
        col += 1
        
        # P3 = u0*v01 -> w1
        U[u_idx[0]][col] = 1; V[v_idx[1]][col] = 1
        W[w_idx[1]][col] = 1
        col += 1
        
        # P4 = u1*v11 -> w1
        U[u_idx[1]][col] = 1; V[v_idx[3]][col] = 1
        W[w_idx[1]][col] = 1
        col += 1

    # Indices
    # A (3x4)
    # Top (2x4): Rows 0,1
    # TL (2x2): [0, 1, 4, 5]
    # TR (2x2): [2, 3, 6, 7]
    # Bot (1x4): Row 2
    # BL (1x2): [8, 9]
    # BR (1x2): [10, 11]
    
    # B (4x4)
    # 00 (2x2): [0, 1, 4, 5]
    # 01 (2x2): [2, 3, 6, 7]
    # 10 (2x2): [8, 9, 12, 13]
    # 11 (2x2): [10, 11, 14, 15]
    
    # C (4x3) W[k*n+i]
    # TL (2x2): k=0,1; i=0,1
    # C11(0,0)=0, C12(0,1)=3, C21(1,0)=1, C22(1,1)=4
    # w_idx_TL = [0, 3, 1, 4]
    
    # TR (2x2): k=2,3; i=0,1
    # C11(2,0)=2, C12(2,1)=5, C21(3,0)=3, C22(3,1)=6
    # w_idx_TR = [2, 5, 3, 6]
    
    # BL (1x2): k=0,1; i=2
    # w0(0,2)=6, w1(1,2)=9
    # w_idx_BL = [6, 9]
    
    # BR (1x2): k=2,3; i=2
    # w0(2,2)=8, w1(3,2)=11
    # w_idx_BR = [8, 11]

    # Strassen calls
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4]) # A_TL, B_00 -> C_TL
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4]) # A_TR, B_10 -> C_TL
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [2, 5, 3, 6]) # A_TL, B_01 -> C_TR
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [2, 5, 3, 6]) # A_TR, B_11 -> C_TR

    # Schoolbook calls
    add_schoolbook_1x2x2([8, 9], [0, 1, 4, 5], [6, 9]) # A_BL, B_00 -> C_BL
    add_schoolbook_1x2x2([10, 11], [8, 9, 12, 13], [6, 9]) # A_BR, B_10 -> C_BL
    add_schoolbook_1x2x2([8, 9], [2, 3, 6, 7], [8, 11]) # A_BL, B_01 -> C_BR
    add_schoolbook_1x2x2([10, 11], [10, 11, 14, 15], [8, 11]) # A_BR, B_11 -> C_BR

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
