def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Split p into 5 blocks of size 2
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = (a-b)(g+h)
    # P4 = d(f-g)
    # P5 = (a+c)h
    # P6 = (b-d)(e+g)
    # P7 = (b+c)f
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Indices for U: i*m + j
    # a: 0*2+0=0, b: 0*2+1=1, c: 1*2+0=2, d: 1*2+1=3
    
    # Indices for V: j*p + k
    # e: 0*10+k, f: 0*10+k+1, g: 1*10+k, h: 1*10+k+1
    # Note: In Strassen, B is [[e, f], [g, h]].
    # So e=B00, f=B01, g=B10, h=B11.
    
    # Indices for W: k*n + i
    # C00: k*2+0, C01: (k+1)*2+0, C10: k*2+1, C11: (k+1)*2+1
    
    for q in range(num_blocks):
        k_start = q * 2
        k_end = k_start + 1
        
        # Column offsets for this block
        col_offset = q * 7
        
        # P1: (a+d)(e+h)
        U[0][col_offset] = 1
        U[3][col_offset] = 1
        V[0*10 + k_start][col_offset] = 1
        V[1*10 + k_end][col_offset] = 1
        W[k_start*2 + 0][col_offset] = 1
        W[k_end*2 + 1][col_offset] = 1
        
        # P2: (c+d)e
        U[2][col_offset+1] = 1
        U[3][col_offset+1] = 1
        V[0*10 + k_start][col_offset+1] = 1
        W[k_start*2 + 1][col_offset+1] = 1
        W[k_end*2 + 1][col_offset+1] = 1
        
        # P3: (a-b)(g+h)
        U[0][col_offset+2] = 1
        U[1][col_offset+2] = -1
        V[1*10 + k_start][col_offset+2] = 1
        V[1*10 + k_end][col_offset+2] = 1
        W[k_start*2 + 0][col_offset+2] = 1
        W[k_end*2 + 1][col_offset+2] = 1
        
        # P4: d(f-g)
        U[3][col_offset+3] = 1
        V[0*10 + k_end][col_offset+3] = 1
        V[1*10 + k_start][col_offset+3] = -1
        W[k_start*2 + 0][col_offset+3] = 1
        W[k_end*2 + 1][col_offset+3] = 1
        
        # P5: (a+c)h
        U[0][col_offset+4] = 1
        U[2][col_offset+4] = 1
        V[1*10 + k_end][col_offset+4] = 1
        W[k_start*2 + 0][col_offset+4] = -1
        W[k_end*2 + 1][col_offset+4] = 1
        
        # P6: (b-d)(e+g)
        U[1][col_offset+5] = 1
        U[3][col_offset+5] = -1
        V[0*10 + k_start][col_offset+5] = 1
        V[1*10 + k_start][col_offset+5] = 1
        W[k_end*2 + 1][col_offset+5] = 1
        
        # P7: (b+c)f
        U[1][col_offset+6] = 1
        U[2][col_offset+6] = 1
        V[0*10 + k_end][col_offset+6] = 1
        W[k_start*2 + 0][col_offset+6] = 1
        W[k_end*2 + 1][col_offset+6] = 1

    return {"U": U, "V": V, "W": W}
