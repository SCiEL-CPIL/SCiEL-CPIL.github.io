"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 factors
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, -1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1],
    ]
    W_strassen = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    rank = 8 * 2 * 7  # 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for h in range(2):
                    for s in range(7):
                        # A block rows: ib*2, ib*2+1; cols: jb*2, jb*2+1
                        # B block rows: jb*2, jb*2+1; cols: kb*4+h*2, kb*4+h*2+1
                        # C block rows: ib*2, ib*2+1; cols: kb*4+h*2, kb*4+h*2+1
                        for ii in range(2):
                            i = ib * 2 + ii
                            for jj in range(2):
                                j = jb * 2 + jj
                                U[i * m + j][col] = U_strassen[ii * 2 + jj][s]
                        for jj in range(2):
                            j = jb * 2 + jj
                            for kk in range(2):
                                k = kb * 4 + h * 2 + kk
                                V[j * p + k][col] = V_strassen[jj * 2 + kk][s]
                        for kk in range(2):
                            k = kb * 4 + h * 2 + kk
                            for ii in range(2):
                                i = ib * 2 + ii
                                W[k * n + i][col] = W_strassen[kk * 2 + ii][s]
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
