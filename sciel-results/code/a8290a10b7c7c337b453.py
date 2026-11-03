def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    j_blocks = [(0,1), (2,3)]
    k_blocks = [(0,1), (2,3), (4,5), (6,)]
    
    for ja, jb in j_blocks:
        for k_block in k_blocks:
            if len(k_block) == 2:
                ka, kb = k_block
                # Strassen 2x2x2: 7 terms
                # M1: (A00+A11)(B00+B11) -> C00 + C11
                U[0*4+ja][col] = 1
                U[1*4+jb][col] = 1
                V[ja*7+ka][col] = 1
                V[jb*7+kb][col] = 1
                W[ka*2+0][col] = 1
                W[kb*2+1][col] = 1
                col += 1
                # M2: (A10+A11)B00 -> C10
                U[1*4+ja][col] = 1
                U[1*4+jb][col] = 1
                V[ja*7+ka][col] = 1
                W[ka*2+1][col] = 1
                col += 1
                # M3: A00(B01-B11) -> C01
                U[0*4+ja][col] = 1
                V[ja*7+kb][col] = 1
                V[jb*7+kb][col] = -1
                W[kb*2+0][col] = 1
                col += 1
                # M4: A11(B10-B00) -> C00 + C10
                U[1*4+jb][col] = 1
                V[jb*7+ka][col] = 1
                V[ja*7+ka][col] = -1
                W[ka*2+0][col] = 1
                W[ka*2+1][col] = 1
                col += 1
                # M5: (A00+A01)B11 -> -C00 + C01
                U[0*4+ja][col] = 1
                U[0*4+jb][col] = 1
                V[jb*7+kb][col] = 1
                W[ka*2+0][col] = -1
                W[kb*2+0][col] = 1
                col += 1
                # M6: (A10-A00)(B00+B01) -> C11
                U[1*4+ja][col] = -1
                U[0*4+ja][col] = 1
                V[ja*7+ka][col] = 1
                V[ja*7+kb][col] = 1
                W[kb*2+1][col] = 1
                col += 1
                # M7: (A01-A11)(B10+B11) -> C00 - C11
                U[0*4+jb][col] = 1
                U[1*4+jb][col] = -1
                V[jb*7+ka][col] = 1
                V[jb*7+kb][col] = 1
                W[ka*2+0][col] = 1
                W[kb*2+1][col] = -1
                col += 1
            else:
                kc = k_block[0]
                # Schoolbook 2x2x1: 4 terms
                for i in range(2):
                    for j in (ja, jb):
                        U[i*4+j][col] = 1
                        V[j*7+kc][col] = 1
                        W[kc*2+i][col] = 1
                        col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
