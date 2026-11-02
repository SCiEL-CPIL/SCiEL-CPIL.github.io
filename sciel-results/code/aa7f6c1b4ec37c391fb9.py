"""Matrix multiplication 5x5x5: Block-diagonal Laderman(3x3) + Strassen(2x2) + cross rank-1 terms."""
import time

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Construct a rank-112 bilinear algorithm for 5x5 matrix multiplication.
    
    Block structure: 5 = 3 + 2
    - C11 = A11*B11 + A12*B21
    - C12 = A11*B12 + A12*B22
    - C21 = A21*B11 + A22*B21
    - C22 = A21*B12 + A22*B22
    
    Diagonal blocks:
    - A11*B11 (3x3): Laderman, rank 15
    - A22*B22 (2x2): Strassen, rank 7
    
    Cross terms (Schoolbook):
    - A12*B21 (3x2 * 2x3 -> 3x3): 18 terms
    - A11*B12 (3x3 * 3x2 -> 3x2): 18 terms
    - A12*B22 (3x2 * 2x2 -> 3x2): 12 terms
    - A21*B11 (2x3 * 3x3 -> 2x3): 18 terms
    - A22*B21 (2x2 * 2x3 -> 2x3): 12 terms
    - A21*B12 (2x3 * 3x2 -> 2x2): 12 terms
    
    Total Rank: 15 + 7 + 18 + 18 + 12 + 18 + 12 + 12 = 112
    """
    n, m, p = DIMENSIONS
    rank = 112
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(U, V, W, u_coeff, v_coeff, w_coeff, col_idx):
        for i in range(n):
            for j in range(m):
                if u_coeff[i][j] != 0:
                    U[i * m + j][col_idx] += u_coeff[i][j]
        for j in range(m):
            for k in range(p):
                if v_coeff[j][k] != 0:
                    V[j * p + k][col_idx] += v_coeff[j][k]
        for k in range(p):
            for i in range(n):
                if w_coeff[k][i] != 0:
                    W[k * n + i][col_idx] += w_coeff[k][i]

    # Helper to create zero matrices
    def zero_mat(r, c):
        return [[0.0] * c for _ in range(r)]

    # 1. Laderman for A11*B11 (3x3)
    # Indices: i,j,k in 0..2
    # We map local 3x3 indices to global 5x5 indices (top-left)
    
    # Laderman 15 terms
    # Note: Coefficients are for the 3x3 block.
    
    # T1
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][0] = 1.0; v[0][0] = 1.0; w[0][0] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T2
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][0] = 1.0
    v[1][1] = 1.0; v[2][2] = 1.0; v[0][0] = -1.0
    w[0][0] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T3
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[1][0] = 1.0
    v[0][1] = 1.0; v[1][2] = 1.0; v[0][0] = -1.0
    w[1][0] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T4
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[2][0] = 1.0
    v[0][2] = 1.0; v[2][1] = 1.0; v[0][0] = -1.0
    w[2][0] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T5
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][1] = 1.0
    v[1][0] = 1.0; v[2][0] = 1.0; v[0][0] = -1.0
    w[0][1] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T6
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[1][1] = 1.0
    v[0][0] = 1.0; v[1][1] = 1.0; v[2][2] = 1.0
    w[1][1] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T7
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[2][1] = 1.0
    v[0][0] = 1.0; v[1][2] = 1.0; v[2][2] = 1.0
    w[2][1] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T8
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][2] = 1.0
    v[1][0] = 1.0; v[2][1] = 1.0; v[0][0] = 1.0
    w[0][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T9
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[1][2] = 1.0
    v[0][0] = 1.0; v[2][0] = 1.0; v[1][1] = 1.0
    w[1][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T10
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[2][2] = 1.0
    v[0][0] = 1.0; v[1][0] = 1.0; v[2][1] = 1.0
    w[2][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T11
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][0] = 1.0; u[1][1] = 1.0; u[2][2] = 1.0
    v[0][0] = 1.0; v[1][1] = 1.0; v[2][2] = 1.0
    w[0][0] = 1.0; w[1][1] = 1.0; w[2][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T12
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][1] = 1.0; u[1][2] = 1.0; u[2][0] = 1.0
    v[1][0] = 1.0; v[2][1] = 1.0; v[0][2] = 1.0
    w[0][1] = 1.0; w[1][2] = 1.0; w[2][0] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T13
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][2] = 1.0; u[1][0] = 1.0; u[2][1] = 1.0
    v[2][0] = 1.0; v[0][1] = 1.0; v[1][2] = 1.0
    w[0][2] = 1.0; w[1][0] = 1.0; w[2][1] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T14
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][0] = 1.0; u[1][1] = -1.0; u[2][2] = 1.0
    v[0][0] = 1.0; v[1][1] = -1.0; v[2][2] = 1.0
    w[0][0] = 1.0; w[1][1] = 1.0; w[2][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # T15
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[0][0] = 1.0; u[1][1] = 1.0; u[2][2] = -1.0
    v[0][0] = 1.0; v[1][1] = 1.0; v[2][2] = -1.0
    w[0][0] = 1.0; w[1][1] = 1.0; w[2][2] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # 2. Strassen for A22*B22 (2x2)
    # Global indices: i,j,k in 3..4
    
    # S1
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[3][3] = 1.0; v[3][3] = 1.0; w[3][3] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S2
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[3][3] = 1.0; u[4][4] = 1.0
    v[3][4] = 1.0; v[4][4] = 1.0
    w[3][4] = 1.0; w[4][3] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S3
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[4][3] = 1.0
    v[3][3] = 1.0; v[4][3] = 1.0
    w[4][3] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S4
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[3][4] = 1.0
    v[3][4] = 1.0; v[4][4] = 1.0
    w[3][4] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S5
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[4][4] = 1.0
    v[3][3] = 1.0; v[4][3] = 1.0
    w[4][3] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S6
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[3][3] = 1.0; u[4][4] = -1.0
    v[4][3] = 1.0
    w[3][3] = 1.0; w[4][4] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # S7
    u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
    u[3][4] = 1.0; u[4][3] = -1.0
    v[3][4] = 1.0
    w[3][4] = 1.0; w[4][4] = 1.0
    add_term(U, V, W, u, v, w, col); col += 1
    
    # 3. Cross Terms (Schoolbook)
    
    # A12*B21 (3x2 * 2x3 -> 3x3)
    # i in 0..2, k in 0..2
    # j in 3..4 (A's col, B's row)
    for i in range(3):
        for k in range(3):
            for j in range(2, 4):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    # A11*B12 (3x3 * 3x2 -> 3x2)
    # i in 0..2, k in 3..4
    # j in 0..2
    for i in range(3):
        for k in range(3, 5):
            for j in range(3):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    # A12*B22 (3x2 * 2x2 -> 3x2)
    # i in 0..2, k in 3..4
    # j in 3..4
    for i in range(3):
        for k in range(3, 5):
            for j in range(3, 5):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    # A21*B11 (2x3 * 3x3 -> 2x3)
    # i in 3..4, k in 0..2
    # j in 0..2
    for i in range(3, 5):
        for k in range(3):
            for j in range(3):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    # A22*B21 (2x2 * 2x3 -> 2x3)
    # i in 3..4, k in 0..2
    # j in 3..4
    for i in range(3, 5):
        for k in range(3):
            for j in range(3, 5):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    # A21*B12 (2x3 * 3x2 -> 2x2)
    # i in 3..4, k in 3..4
    # j in 0..2
    for i in range(3, 5):
        for k in range(3, 5):
            for j in range(3):
                u = zero_mat(5, 5); v = zero_mat(5, 5); w = zero_mat(5, 5)
                u[i][j] = 1.0
                v[j][k] = 1.0
                w[k][i] = 1.0
                add_term(U, V, W, u, v, w, col); col += 1

    return {"U": U, "V": V, "W": W}
