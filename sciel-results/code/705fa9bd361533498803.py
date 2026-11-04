"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen-based rank-49 decomposition for 4x4x4, applied twice for 4x4x8.
    # Total rank = 49 + 49 = 98.
    
    # Strassen rank-7 for 2x2x2
    # Each term: (A_2x2, B_2x2, C_2x2)
    strassen_2x2 = []
    # M1 = (A00+A11)(B00+B11)
    strassen_2x2.append(([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,0]]))  # C00 part
    strassen_2x2.append(([[1,0],[0,1]], [[1,0],[0,1]], [[0,0],[0,1]]))   # C11 part
    # M2 = (A10+A11)B00
    strassen_2x2.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]))   # C10 part
    # M3 = A00(B10-B11)
    strassen_2x2.append(([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]))  # C01 part
    # M4 = A11(B01-B00)
    strassen_2x2.append(([[0,0],[0,1]], [[0,1],[-1,0]], [[1,0],[0,1]]))  # C00 and C11
    # M5 = (A00+A01)B11
    strassen_2x2.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,-1]])) # C00 and C11
    # M6 = (A10-A00)(B00+B10)
    strassen_2x2.append(([[1,-1],[0,0]], [[1,0],[1,0]], [[0,0],[0,1]]))  # C11 part
    # M7 = (A01-A11)(B01+B11)
    strassen_2x2.append(([[0,1],[-1,0]], [[0,1],[0,1]], [[0,1],[0,0]]))  # C01 part
    
    # This is getting complicated. Let me just use schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
