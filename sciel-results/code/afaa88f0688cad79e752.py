def construct():
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 2x2 x 2x2 decomposition applied to 4 blocks
    for b in range(4):
        base = b * 7
        
        # M1 = (A00+A11)*(B00+B11)
        # A00 is U[0], A11 is U[3]
        # B00 of block b is V[0*p + 2*b]
        # B11 of block b is V[1*p + 2*b + 1]
        # C00 of block b is W[(2*b)*n + 0]
        # C11 of block b is W[(2*b+1)*n + 1]
        U[0][base+0] += 1
        U[3][base+0] += 1
        V[0*p + 2*b][base+0] += 1
        V[1*p + 2*b + 1][base+0] += 1
        W[(2*b)*n + 0][base+0] += 1
        W[(2*b+1)*n + 1][base+0] += 1
        
        # M2 = (A10+A11)*B00
        # A10 is U[1]
        # C10 of block b is W[(2*b)*n + 1]
        U[1][base+1] += 1
        U[3][base+1] += 1
        V[0*p + 2*b][base+1] += 1
        W[(2*b)*n + 1][base+1] += 1
        
        # M3 = A00*(B01-B11)
        # B01 of block b is V[0*p + 2*b + 1]
        # B11 of block b is V[1*p + 2*b + 1]
        # C01 of block b is W[(2*b+1)*n + 0]
        U[0][base+2] += 1
        V[0*p + 2*b + 1][base+2] += 1
        V[1*p + 2*b + 1][base+2] -= 1
        W[(2*b+1)*n + 0][base+2] += 1
        
        # M4 = A11*(B10-B00)
        # B10 of block b is V[1*p + 2*b]
        # B00 of block b is V[0*p + 2*b]
        # C00 of block b is W[(2*b)*n + 0] (coefficient -1)
        # C10 of block b is W[(2*b)*n + 1] (coefficient +1)
        U[3][base+3] += 1
        V[1*p + 2*b][base+3] += 1
        V[0*p + 2*b][base+3] -= 1
        W[(2*b)*n + 0][base+3] -= 1
        W[(2*b)*n + 1][base+3] += 1
        
        # M5 = (A00+A01)*B11
        # A01 is U[2]
        # C00 of block b is W[(2*b)*n + 0] (coefficient -1)
        # C01 of block b is W[(2*b+1)*n + 0] (coefficient +1)
        U[0][base+4] += 1
        U[2][base+4] += 1
        V[1*p + 2*b + 1][base+4] += 1
        W[(2*b)*n + 0][base+4] -= 1
        W[(2*b+1)*n + 0][base+4] += 1
        
        # M6 = (A10-A00)*(B00+B10)
        # C11 of block b is W[(2*b+1)*n + 1] (coefficient -1)
        U[1][base+5] += 1
        U[0][base+5] -= 1
        V[0*p + 2*b][base+5] += 1
        V[1*p + 2*b][base+5] += 1
        W[(2*b+1)*n + 1][base+5] -= 1
        
        # M7 = (A01-A11)*(B01+B11)
        # C00 of block b is W[(2*b)*n + 0] (coefficient +1)
        # C11 of block b is W[(2*b+1)*n + 1] (coefficient +1)
        U[2][base+6] += 1
        U[3][base+6] -= 1
        V[0*p + 2*b + 1][base+6] += 1
        V[1*p + 2*b + 1][base+6] += 1
        W[(2*b)*n + 0][base+6] += 1
        W[(2*b+1)*n + 1][base+6] += 1
        
    return {"U": U, "V": V, "W": W}
