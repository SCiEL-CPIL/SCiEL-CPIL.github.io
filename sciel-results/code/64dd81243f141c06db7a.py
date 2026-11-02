import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1

    # Helper to add schoolbook 2x2x1
    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b10]
        # c_idx: [c00, c10]
        
        # P1: a00 * b00 -> c00
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # P2: a01 * b10 -> c00
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # P3: a10 * b00 -> c10
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # P4: a11 * b10 -> c10
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1

    # Define indices
    # A: 2x4
    # A1: cols 0,1. Indices: 0,1,4,5
    a1_idx = [0, 1, 4, 5]
    # A2: cols 2,3. Indices: 2,3,6,7
    a2_idx = [2, 3, 6, 7]
    
    # B: 4x5
    # B11: rows 0,1 cols 0,1. Indices: 0,1,5,6
    b11_idx = [0, 1, 5, 6]
    # B21: rows 2,3 cols 0,1. Indices: 10,11,15,16
    b21_idx = [10, 11, 15, 16]
    # B12: rows 0,1 cols 2,3. Indices: 2,3,7,8
    b12_idx = [2, 3, 7, 8]
    # B22: rows 2,3 cols 2,3. Indices: 12,13,17,18
    b22_idx = [12, 13, 17, 18]
    # B13: rows 0,1 col 4. Indices: 4, 9
    b13_idx = [4, 9]
    # B23: rows 2,3 col 4. Indices: 14, 19
    b23_idx = [14, 19]
    
    # C: 2x5 (stored as 5x2 in W)
    # C cols 0,1: k=0,1.
    # c00 (i=0,k=0) -> 0
    # c01 (i=0,k=1) -> 2
    # c10 (i=1,k=0) -> 1
    # c11 (i=1,k=1) -> 3
    c_01_idx = [0, 2, 1, 3]
    
    # C cols 2,3: k=2,3.
    # c00 (i=0,k=2) -> 4
    # c01 (i=0,k=3) -> 6
    # c10 (i=1,k=2) -> 5
    # c11 (i=1,k=3) -> 7
    c_23_idx = [4, 6, 5, 7]
    
    # C col 4: k=4.
    # c00 (i=0,k=4) -> 8
    # c10 (i=1,k=4) -> 9
    c_4_idx = [8, 9]

    # Execute decompositions
    add_strassen(a1_idx, b11_idx, c_01_idx)
    add_strassen(a2_idx, b21_idx, c_01_idx)
    add_strassen(a1_idx, b12_idx, c_23_idx)
    add_strassen(a2_idx, b22_idx, c_23_idx)
    add_schoolbook_2x2x1(a1_idx, b13_idx, c_4_idx)
    add_schoolbook_2x2x1(a2_idx, b23_idx, c_4_idx)
    
    return {"U": U, "V": V, "W": W}
