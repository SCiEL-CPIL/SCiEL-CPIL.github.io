"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    # 0-indexed splits
    i_blocks = [(0, 1), (2,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    # Compute total rank
    rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni, nj, nk = len(ib), len(jb), len(kb)
                if ni == 2 and nj == 2 and nk == 2:
                    rank += 7
                else:
                    rank += ni * nj * nk
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni, nj, nk = len(ib), len(jb), len(kb)
                
                if ni == 2 and nj == 2 and nk == 2:
                    i0, i1 = ib
                    j0, j1 = jb
                    k0, k1 = kb
                    
                    # M1 = (A00+A11)(B00+B11)
                    U[i0*m+j0][col] += 1
                    U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1
                    V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1
                    W[k1*n+i1][col] += 1
                    col += 1
                    
                    # M2 = (A10+A11)B00
                    U[i1*m+j0][col] += 1
                    U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1
                    W[k0*n+i1][col] += 1
                    col += 1
                    
                    # M3 = A00(B10-B00)
                    U[i0*m+j0][col] += 1
                    V[j1*p+k0][col] += 1
                    V[j0*p+k0][col] -= 1
                    W[k1*n+i0][col] += 1
                    col += 1
                    
                    # M4 = A11(B01-B11)
                    U[i1*m+j1][col] += 1
                    V[j0*p+k1][col] += 1
                    V[j1*p+k1][col] -= 1
                    W[k1*n+i1][col] += 1
                    col += 1
                    
                    # M5 = (A00+A01)B11
                    U[i0*m+j0][col] += 1
                    U[i0*m+j1][col] += 1
                    V[j1*p+k1][col] += 1
                    W[k1*n+i0][col] -= 1
                    col += 1
                    
                    # M6 = (A10-A00)(B00+B01)
                    U[i1*m+j0][col] += 1
                    U[i0*m+j0][col] -= 1
                    V[j0*p+k0][col] += 1
                    V[j0*p+k1][col] += 1
                    W[k1*n+i1][col] += 1
                    col += 1
                    
                    # M7 = (A01-A11)(B10+B11)
                    U[i0*m+j1][col] += 1
                    U[i1*m+j1][col] -= 1
                    V[j1*p+k0][col] += 1
                    V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1
                    col += 1
                    
                else:
                    for ii in ib:
                        for jj in jb:
                            for kk in kb:
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
