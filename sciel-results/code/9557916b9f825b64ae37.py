"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Try rank 28 first
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[i0*m+j0][col] = 1
        U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1
        W[(k0+1)*n+i0][col] = 1
        W[k0*n+(i0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M2 = (a10+a11)b00
        U[(i0+1)*m+j0][col] = 1
        U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+(i0+1)][col] = 1
        col += 1
        # M3 = a00(b10-b11)
        U[i0*m+j0][col] = 1
        V[(j0+1)*p+k0][col] = 1
        V[(j0+1)*p+(k0+1)][col] = -1
        W[k0*n+i0][col] = 1
        W[(k0+1)*n+i0][col] = 1
        col += 1
        # M4 = a11(b01-b00)
        U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+(k0+1)][col] = 1
        V[j0*p+k0][col] = -1
        W[k0*n+(i0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M5 = (a00+a01)b11
        U[i0*m+j0][col] = 1
        U[i0*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = -1
        W[(k0+1)*n+i0][col] = 1
        col += 1
        # M6 = (a10-a00)(b00+b10)
        U[(i0+1)*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[(j0+1)*p+k0][col] = 1
        W[k0*n+(i0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        # M7 = (a01-a11)(b01+b11)
        U[i0*m+(j0+1)][col] = 1
        U[(i0+1)*m+(j0+1)][col] = -1
        V[j0*p+(k0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1
        W[(k0+1)*n+i0][col] = -1
        col += 1
    
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    
    # Schoolbook for remaining
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    for i in range(2):
        for k in range(5):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    assert col == 28
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    # Fall back to schoolbook rank 30
                    rank30 = n * m * p
                    U30 = [[0] * rank30 for _ in range(n * m)]
                    V30 = [[0] * rank30 for _ in range(m * p)]
                    W30 = [[0] * rank30 for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U30[ii*m+jj][c] = 1
                                V30[jj*p+kk][c] = 1
                                W30[kk*n+ii][c] = 1
                                c += 1
                    return {"U": U30, "V": V30, "W": W30}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
