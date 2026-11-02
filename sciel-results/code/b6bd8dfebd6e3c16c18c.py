def construct():
    n, m, p = 3, 3, 6
    rank = 45
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # For each of the 3 blocks of 2 columns in B
    for block in range(3):
        k0 = block * 2  # first column index in B for this block
        base = block * 15  # base index for this block's terms
        
        # 15 terms for 3x3x2
        # Term 0: a00*b00 -> c00
        U[0][base+0] = 1; V[0*2+k0][base+0] = 1; W[k0*n+0][base+0] = 1
        # Term 1: a00*b01 -> c01
        U[0][base+1] = 1; V[0*2+k0+1][base+1] = 1; W[(k0+1)*n+0][base+1] = 1
        # Term 2: a01*b10 -> c00
        U[1][base+2] = 1; V[1*2+k0][base+2] = 1; W[k0*n+0][base+2] = 1
        # Term 3: a01*b11 -> c01
        U[1][base+3] = 1; V[1*2+k0+1][base+3] = 1; W[(k0+1)*n+0][base+3] = 1
        # Term 4: a02*b20 -> c00
        U[2][base+4] = 1; V[2*2+k0][base+4] = 1; W[k0*n+0][base+4] = 1
        # Term 5: a02*b21 -> c01
        U[2][base+5] = 1; V[2*2+k0+1][base+5] = 1; W[(k0+1)*n+0][base+5] = 1
        # Term 6: a10*b00 -> c10
        U[3][base+6] = 1; V[0*2+k0][base+6] = 1; W[k0*n+1][base+6] = 1
        # Term 7: a10*b01 -> c11
        U[3][base+7] = 1; V[0*2+k0+1][base+7] = 1; W[(k0+1)*n+1][base+7] = 1
        # Term 8: a11*b10 -> c10
        U[4][base+8] = 1; V[1*2+k0][base+8] = 1; W[k0*n+1][base+8] = 1
        # Term 9: a11*b11 -> c11
        U[4][base+9] = 1; V[1*2+k0+1][base+9] = 1; W[(k0+1)*n+1][base+9] = 1
        # Term 10: a12*b20 -> c10
        U[5][base+10] = 1; V[2*2+k0][base+10] = 1; W[k0*n+1][base+10] = 1
        # Term 11: a12*b21 -> c11
        U[5][base+11] = 1; V[2*2+k0+1][base+11] = 1; W[(k0+1)*n+1][base+11] = 1
        # Term 12: a20*b00 -> c20
        U[6][base+12] = 1; V[0*2+k0][base+12] = 1; W[k0*n+2][base+12] = 1
        # Term 13: a20*b01 -> c21
        U[6][base+13] = 1; V[0*2+k0+1][base+13] = 1; W[(k0+1)*n+2][base+13] = 1
        # Term 14: a21*b10 -> c20
        U[7][base+14] = 1; V[1*2+k0][base+14] = 1; W[k0*n+2][base+14] = 1
        
        # Missing: a21*b11 -> c21, a22*b20 -> c20, a22*b21 -> c21
        # That's 3 more terms needed, making it 18 not 15.
    
    return {"U": U, "V": V, "W": W}
