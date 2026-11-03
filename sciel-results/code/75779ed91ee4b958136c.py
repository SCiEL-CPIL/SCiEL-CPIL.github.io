def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen helper
    def add_strassen(a_idx, b_idx, c_idx, col_offset):
        # a_idx: list of 4 global U indices for a,b,c,d
        # b_idx: list of 4 global V indices for e,f,g,h
        # c_idx: list of 4 global W indices for p,q,r,s
        # col_offset: start column
        
        # m1 = (a+d)(e+h)
        U[a_idx[0]][col_offset] += 1
        U[a_idx[3]][col_offset] += 1
        V[b_idx[0]][col_offset] += 1
        V[b_idx[3]][col_offset] += 1
        W[c_idx[0]][col_offset] += 1 # p
        W[c_idx[3]][col_offset] += 1 # s
        
        # m2 = (c+d)e
        U[a_idx[2]][col_offset+1] += 1
        U[a_idx[3]][col_offset+1] += 1
        V[b_idx[0]][col_offset+1] += 1
        W[c_idx[2]][col_offset+1] += 1 # r
        W[c_idx[3]][col_offset+1] -= 1 # s
        
        # m3 = a(f-h)
        U[a_idx[0]][col_offset+2] += 1
        V[b_idx[1]][col_offset+2] += 1
        V[b_idx[3]][col_offset+2] -= 1
        W[c_idx[1]][col_offset+2] += 1 # q
        W[c_idx[3]][col_offset+2] += 1 # s
        
        # m4 = d(g-e)
        U[a_idx[3]][col_offset+3] += 1
        V[b_idx[2]][col_offset+3] += 1
        V[b_idx[0]][col_offset+3] -= 1
        W[c_idx[0]][col_offset+3] += 1 # p
        W[c_idx[2]][col_offset+3] += 1 # r
        
        # m5 = (a+b)h
        U[a_idx[0]][col_offset+4] += 1
        U[a_idx[1]][col_offset+4] += 1
        V[b_idx[3]][col_offset+4] += 1
        W[c_idx[0]][col_offset+4] -= 1 # p
        W[c_idx[1]][col_offset+4] += 1 # q
        
        # m6 = (c-a)(e+f)
        U[a_idx[2]][col_offset+5] += 1
        U[a_idx[0]][col_offset+5] -= 1
        V[b_idx[0]][col_offset+5] += 1
        V[b_idx[1]][col_offset+5] += 1
        W[c_idx[3]][col_offset+5] += 1 # s
        
        # m7 = (b-d)(g+h)
        U[a_idx[1]][col_offset+6] += 1
        U[a_idx[3]][col_offset+6] -= 1
        V[b_idx[2]][col_offset+6] += 1
        V[b_idx[3]][col_offset+6] += 1
        W[c_idx[0]][col_offset+6] += 1 # p
        # Note: m7 also contributes to p?
        # p = m1 + m4 - m5 + m7. Yes.
        
    # Build 2x2x6
    def add_2x2x6(a_idx, b_idx, c_idx, col_offset):
        # 3 blocks of 2x2x2
        # Block 1: B cols 0-1, C cols 0-1
        b1 = [b_idx[0], b_idx[1], b_idx[6], b_idx[7]] # e,f,g,h for B[0:2, 0:2]
        c1 = [c_idx[0], c_idx[1], c_idx[6], c_idx[7]] # p,q,r,s for C[0:2, 0:2]
        add_strassen(a_idx, b1, c1, col_offset)
        
        # Block 2: B cols 2-3, C cols 2-3
        b2 = [b_idx[2], b_idx[3], b_idx[8], b_idx[9]]
        c2 = [c_idx[2], c_idx[3], c_idx[8], c_idx[9]]
        add_strassen(a_idx, b2, c2, col_offset + 7)
        
        # Block 3: B cols 4-5, C cols 4-5
        b3 = [b_idx[4], b_idx[5], b_idx[10], b_idx[11]]
        c3 = [c_idx[4], c_idx[5], c_idx[10], c_idx[11]]
        add_strassen(a_idx, b3, c3, col_offset + 14)

    # Build 2x4x6
    # A1: indices 0,1,4,5
    # B1: indices 0..11
    # C: 0..11
    a1 = [0, 1, 4, 5]
    b1 = list(range(0, 12))
    c1 = list(range(0, 12))
    add_2x2x6(a1, b1, c1, 0)
    
    # A2: indices 2,3,6,7
    # B2: indices 12..23
    # C: 0..11 (same C)
    a2 = [2, 3, 6, 7]
    b2 = list(range(12, 24))
    c2 = list(range(0, 12))
    add_2x2x6(a2, b2, c2, 21)
    
    return {"U": U, "V": V, "W": W}
