DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # A indices: (i1, j1), (i1, j2), (i2, j1), (i2, j2)
    # B indices: (j1, k1), (j1, k2), (j2, k1), (j2, k2)
    # C indices: (i1, k1), (i1, k2), (i2, k1), (i2, k2)
    def add_strassen(start_col, a_i1, a_j1, a_i2, a_j2, b_j1, b_j2, b_k1, b_k2, c_i1, c_i2, c_k1, c_k2):
        # Map global indices
        # A: i*m + j
        # B: j*p + k
        # C: k*n + i
        
        a00 = a_i1 * m + a_j1
        a01 = a_i1 * m + a_j2
        a10 = a_i2 * m + a_j1
        a11 = a_i2 * m + a_j2
        
        b00 = b_j1 * p + b_k1
        b01 = b_j1 * p + b_k2
        b10 = b_j2 * p + b_k1
        b11 = b_j2 * p + b_k2
        
        c00 = c_k1 * n + c_i1
        c01 = c_k2 * n + c_i1
        c10 = c_k1 * n + c_i2
        c11 = c_k2 * n + c_i2
        
        # M1 = (a00 + a11)(b00 + b11)
        U[a00][start_col] += 1
        U[a11][start_col] += 1
        V[b00][start_col] += 1
        V[b11][start_col] += 1
        W[c00][start_col] += 1
        W[c11][start_col] += 1
        
        # M2 = (a10 + a11)b00
        U[a10][start_col+1] += 1
        U[a11][start_col+1] += 1
        V[b00][start_col+1] += 1
        W[c10][start_col+1] += 1
        W[c11][start_col+1] -= 1
        
        # M3 = a00(b01 - b11)
        U[a00][start_col+2] += 1
        V[b01][start_col+2] += 1
        V[b11][start_col+2] -= 1
        W[c01][start_col+2] += 1
        W[c11][start_col+2] += 1
        
        # M4 = a11(b10 - b00)
        U[a11][start_col+3] += 1
        V[b10][start_col+3] += 1
        V[b00][start_col+3] -= 1
        W[c00][start_col+3] += 1
        W[c10][start_col+3] += 1
        
        # M5 = (a00 + a01)b11
        U[a00][start_col+4] += 1
        U[a01][start_col+4] += 1
        V[b11][start_col+4] += 1
        W[c00][start_col+4] -= 1
        W[c01][start_col+4] += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10][start_col+5] += 1
        U[a00][start_col+5] -= 1
        V[b00][start_col+5] += 1
        V[b01][start_col+5] += 1
        W[c11][start_col+5] += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01][start_col+6] += 1
        U[a11][start_col+6] -= 1
        V[b10][start_col+6] += 1
        V[b11][start_col+6] += 1
        W[c00][start_col+6] += 1
        W[c11][start_col+6] -= 1

    # Helper to add Trivial block 2x2x1
    # A: 2x2, B: 2x1, C: 2x1
    def add_trivial_221(start_col, a_i1, a_j1, a_i2, a_j2, b_j1, b_j2, b_k, c_i1, c_i2):
        a00 = a_i1 * m + a_j1
        a01 = a_i1 * m + a_j2
        a10 = a_i2 * m + a_j1
        a11 = a_i2 * m + a_j2
        
        b00 = b_j1 * p + b_k
        b10 = b_j2 * p + b_k
        
        c00 = b_k * n + c_i1
        c10 = b_k * n + c_i2
        
        # P1 = a00 b00
        U[a00][start_col] += 1
        V[b00][start_col] += 1
        W[c00][start_col] += 1
        
        # P2 = a01 b10
        U[a01][start_col+1] += 1
        V[b10][start_col+1] += 1
        W[c00][start_col+1] += 1
        
        # P3 = a10 b00
        U[a10][start_col+2] += 1
        V[b00][start_col+2] += 1
        W[c10][start_col+2] += 1
        
        # P4 = a11 b10
        U[a11][start_col+3] += 1
        V[b10][start_col+3] += 1
        W[c10][start_col+3] += 1

    # Block 1: A_L (i=0,1, j=0,1) * B_T (j=0,1, k=0,1)
    # A_L: i1=0, i2=1, j1=0, j2=1
    # B_T1: j1=0, j2=1, k1=0, k2=1
    add_strassen(0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A_L * B_T2 (j=0,1, k=2,3)
    add_strassen(7, 0, 0, 1, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    
    # Block 3: A_L * B_T3 (j=0,1, k=4)
    add_trivial_221(14, 0, 0, 1, 1, 0, 1, 4, 0, 1)
    
    # Block 4: A_R (i=0,1, j=2,3) * B_B (j=2,3, k=0,1)
    add_strassen(18, 0, 2, 1, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    
    # Block 5: A_R * B_B2 (j=2,3, k=2,3)
    add_strassen(25, 0, 2, 1, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # Block 6: A_R * B_B3 (j=2,3, k=4)
    add_trivial_221(32, 0, 2, 1, 3, 2, 3, 4, 0, 1)
    
    return {"U": U, "V": V, "W": W}
