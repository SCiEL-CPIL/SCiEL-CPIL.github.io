"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 rank-7 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    su = [[1,1],[0,1],[1,0],[0,1],[1,1],[-1,1],[1,-1]]
    sv = [[1,1],[1,1],[1,0],[0,1],[1,0],[1,0],[0,1]]
    sw = [[1,1],[1,0],[1,-1],[1,-1],[0,1],[1,1],[1,1]]
    
    col = 0
    for jb in (0, 2):
        for kb in (0, 2):
            # Strassen for i in {0,1}
            for c in range(7):
                for i in range(2):
                    for dj in range(2):
                        U[i * m + jb + dj][col] = su[c][i]
                for dj in range(2):
                    for dk in range(2):
                        V[(jb + dj) * p + kb + dk][col] = sv[c][dj]
                for dk in range(2):
                    for i in range(2):
                        W[(kb + dk) * n + i][col] = sw[c][dk]
                col += 1
            # Schoolbook for i=2
            for dj in range(2):
                for dk in range(2):
                    U[2 * m + jb + dj][col] = 1
                    V[(jb + dj) * p + kb + dk][col] = 1
                    W[(kb + dk) * n + 2][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
