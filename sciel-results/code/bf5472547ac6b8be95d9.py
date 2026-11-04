"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Strassen coefficients for 2x2x2
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # m1 = (A11 - A22) * (B22 - B11)
    # m2 = (A11 + A22) * B11
    # m3 = A22 * (B12 - B11)
    # m4 = A11 * (B22 - B21)
    # m5 = (A11 + A12) * B21
    # m6 = (A21 - A11) * (B11 + B22)
    # m7 = (A12 - A22) * (B21 + B22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U: rows are (i, j) for A[i,j], j is the middle index
    # U[i*m+j, t]
    strassen_U = [
        [1, 1, 0, 1, 1, -1, 0],    # A11 (i=0, j=0)
        [0, 0, 0, 0, 1, 0, 1],     # A12 (i=0, j=1)
        [-1, 0, 0, 0, 0, 1, 0],    # A21 (i=1, j=0)
        [-1, 1, 1, 0, 0, 0, -1]    # A22 (i=1, j=1)
    ]
    
    # V: rows are (j, k) for B[j,k]
    strassen_V = [
        [-1, 1, -1, 0, 0, 1, 0],   # B11 (j=0, k=0)
        [0, 0, 1, 0, 0, 0, 1],     # B12 (j=0, k=1)
        [-1, 0, 0, -1, 1, 0, 1],   # B21 (j=1, k=0)
        [1, 0, 0, 1, 0, 0, 0]      # B22 (j=1, k=1)
    ]
    
    # W: rows are (k, i) for C[k*n+i]
    strassen_W = [
        [1, 0, 0, 0, 0, 0, 1],     # C11 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],     # C12 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],     # C21 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]     # C22 (k=1, i=1)
    ]
    
    # Total rank: 3*(7 + 4) + 6 = 39
    rank_total = 39
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0, col_start):
        for t in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di) * m + (j0 + dj)][col_start + t] = strassen_U[di * 2 + dj][t]
            for dj in range(2):
                for dk in range(2):
                    V[(j0 + dj) * p + (k0 + dk)][col_start + t] = strassen_V[dj * 2 + dk][t]
            for dk in range(2):
                for di in range(2):
                    W[(k0 + dk) * n + (i0 + di)][col_start + t] = strassen_W[dk * 2 + di][t]
    
    def add_direct_2x1x2(i0, j0, k0, col_start):
        for dk in range(2):
            for di in range(2):
                U[(i0 + di) * m + j0][col_start + dk] = 1
            V[j0 * p + (k0 + dk)][col_start + dk] = 1
            for di in range(2):
                W[(k0 + dk) * n + (i0 + di)][col_start + dk] = 1
    
    def add_direct_2x3x1(i0, j0, k0, col_start):
        for dj in range(3):
            for di in range(2):
                U[(i0 + di) * m + (j0 + dj)][col_start + dj] = 1
            V[(j0 + dj) * p + k0][col_start + dj] = 1
            for di in range(2):
                W[k0 * n + (i0 + di)][col_start + dj] = 1
    
    # Block 1: B[:, 0:2]
    add_strassen(0, 0, 0, col)
    col += 7
    add_direct_2x1x2(0, 2, 0, col)
    col += 4
    
    # Block 2: B[:, 2:4]
    add_strassen(0, 0, 2, col)
    col += 7
    add_direct_2x1x2(0, 2, 2, col)
    col += 4
    
    # Block 3: B[:, 4:6]
    add_strassen(0, 0, 4, col)
    col += 7
    add_direct_2x1x2(0, 2, 4, col)
    col += 4
    
    # Block 4: B[:, 6:7]
    add_direct_2x3x1(0, 0, 6, col)
    col += 3
    
    assert col == rank_total, f"Expected {rank_total} but got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
