DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices for A: 0:a, 1:b, 2:c, 3:d
    # V indices for B block: 0:e, 1:f, 2:g, 3:h (offset by 2*block)
    # W indices for C block: 0:C00, 1:C01, 2:C10, 3:C11 (offset by 2*block)
    
    # Coefficients for U (A part)
    # M1: a, d -> 1, 1
    # M2: c, d -> 1, 1
    # M3: a -> 1
    # M4: d -> 1
    # M5: a, b -> 1, 1
    # M6: c, a -> 1, -1
    # M7: b, d -> 1, -1
    
    u_coeffs = [
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # This is wrong, need 7 cols per block
    ]
    
    # Let's build it block by block
    for b in range(3):
        base_u = 0
        base_v = 0
        base_w = 0
        offset_v = b * 2
        offset_w = b * 2
        
        # M1
        col = b * 7
        U[0][col] += 1 # a
        U[3][col] += 1 # d
        V[offset_v][col] += 1 # e
        V[offset_v+1][col] += 1 # f? No, M1=(a+d)(e+h). h is V[offset_v+3]
        # Correction: V indices for e,f,g,h are 0,1,2,3 relative to block start
        # V row 0: e, f. V row 1: g, h.
        # So e=V[0], f=V[1], g=V[2], h=V[3] relative to block.
        # Absolute V index: row * 6 + col.
        # Block b starts at col 2*b.
        # e is at V[0*6 + 2*b]
        # f is at V[0*6 + 2*b + 1]
        # g is at V[1*6 + 2*b]
        # h is at V[1*6 + 2*b + 1]
        
        # M1 = (a+d)(e+h)
        U[0][col] = 1
        U[3][col] = 1
        V[0*6 + 2*b][col] = 1
        V[1*6 + 2*b + 1][col] = 1
        # C11 += M1, C22 += M1
        W[2*b*2 + 0][col] = 1
        W[(2*b+1)*2 + 1][col] = 1
        
        # M2 = (c+d)e
        col += 1
        U[2][col] = 1
        U[3][col] = 1
        V[0*6 + 2*b][col] = 1
        # C21 += M2
        W[2*b*2 + 2][col] = 1
        
        # M3 = a(f-h)
        col += 1
        U[0][col] = 1
        V[0*6 + 2*b + 1][col] = 1
        V[1*6 + 2*b + 1][col] = -1
        # C12 += M3, C22 += M3
        W[2*b*2 + 1][col] = 1
        W[(2*b+1)*2 + 1][col] = 1
        
        # M4 = d(g-f)
        col += 1
        U[3][col] = 1
        V[1*6 + 2*b][col] = 1
        V[0*6 + 2*b + 1][col] = -1
        # C11 += M4, C21 += M4
        W[2*b*2 + 0][col] = 1
        W[2*b*2 + 2][col] = 1
        
        # M5 = (a+b)h
        col += 1
        U[0][col] = 1
        U[1][col] = 1
        V[1*6 + 2*b + 1][col] = 1
        # C11 -= M5, C12 += M5
        W[2*b*2 + 0][col] = -1
        W[2*b*2 + 1][col] = 1
        
        # M6 = (c-a)(e+f)
        col += 1
        U[2][col] = 1
        U[0][col] = -1
        V[0*6 + 2*b][col] = 1
        V[0*6 + 2*b + 1][col] = 1
        # C22 += M6
        W[(2*b+1)*2 + 1][col] = 1
        
        # M7 = (b-d)(g+h)
        col += 1
        U[1][col] = 1
        U[3][col] = -1
        V[1*6 + 2*b][col] = 1
        V[1*6 + 2*b + 1][col] = 1
        # C11 += M7
        W[2*b*2 + 0][col] = 1
        
    return {"U": U, "V": V, "W": W}
