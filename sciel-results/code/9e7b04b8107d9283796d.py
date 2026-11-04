# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Part 1: 3x3 * 3x3 (cols 0-2 of B, cols 0-2 of C)
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # a00,a01,a10,a11 and b00,b01,b10,b11
    # M1 = (a00+a11)(b00+b11)
    add_term(0, 0, 0, 1, 1, 1, col) # a00 b00
    add_term(1, 1, 0, 1, 1, 1, col) # a11 b00? No.
    # Let's use explicit terms for Strassen to be safe
    # M1 = (a00+a11)(b00+b11)
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    # M2 = (a00+a11)(b10-b00)
    U[0][col] = 1; U[1][col] = 1
    V[2][col] = 1; V[3][col] = -1
    W[0][col] = -1; W[1][col] = -1; W[3][col] = 1; W[4][col] = 1
    col += 1
    # M3 = a00(b10-b11)
    U[0][col] = 1
    V[2][col] = 1; V[3][col] = -1
    W[0][col] = -1; W[1][col] = -1
    col += 1
    # M4 = a11(b00-b01)
    U[1][col] = 1
    V[0][col] = 1; V[1][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    # M5 = (a00+a01)b11
    U[0][col] = 1; U[2][col] = 1
    V[3][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # M6 = (a10-a00)(b00+b01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = -1; W[4][col] = -1
    col += 1
    # M7 = (a01-a11)(b10+b11)
    U[2][col] = 1; U[1][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Cross terms for 3x3x3
    # A12 B21: a02 b20, a02 b21, a12 b20, a12 b21
    add_term(2, 6, 0, 1, 1, 1, col); col += 1 # a02 b20 -> c00
    add_term(2, 7, 1, 1, 1, 1, col); col += 1 # a02 b21 -> c01
    add_term(5, 6, 3, 1, 1, 1, col); col += 1 # a12 b20 -> c10
    add_term(5, 7, 4, 1, 1, 1, col); col += 1 # a12 b21 -> c11
    
    # A11 B12: a00 b02, a01 b12, a10 b02, a11 b12
    add_term(0, 4, 6, 1, 1, 1, col); col += 1 # a00 b02 -> c02
    add_term(2, 5, 6, 1, 1, 1, col); col += 1 # a01 b12 -> c02
    add_term(3, 4, 9, 1, 1, 1, col); col += 1 # a10 b02 -> c12
    add_term(5, 5, 9, 1, 1, 1, col); col += 1 # a11 b12 -> c12
    
    # A12 B22: a02 b22, a12 b22
    add_term(2, 8, 6, 1, 1, 1, col); col += 1 # a02 b22 -> c02
    add_term(5, 8, 9, 1, 1, 1, col); col += 1 # a12 b22 -> c12
    
    # A21 B11: a20 b00, a20 b10, a21 b00, a21 b10
    add_term(6, 0, 12, 1, 1, 1, col); col += 1 # a20 b00 -> c20
    add_term(6, 2, 13, 1, 1, 1, col); col += 1 # a20 b10 -> c21
    add_term(7, 0, 12, 1, 1, 1, col); col += 1 # a21 b00 -> c20
    add_term(7, 2, 13, 1, 1, 1, col); col += 1 # a21 b10 -> c21
    
    # A21 B12: a20 b02, a21 b12
    add_term(6, 4, 15, 1, 1, 1, col); col += 1 # a20 b02 -> c22
    add_term(7, 5, 15, 1, 1, 1, col); col += 1 # a21 b12 -> c22
    
    # A22 B21: a22 b20, a22 b21
    add_term(8, 6, 12, 1, 1, 1, col); col += 1 # a22 b20 -> c20
    add_term(8, 7, 13, 1, 1, 1, col); col += 1 # a22 b21 -> c21
    
    # A22 B22: a22 b22
    add_term(8, 8, 15, 1, 1, 1, col); col += 1 # a22 b22 -> c22
    
    # Part 2: 3x3 * 3x2 (cols 3-4 of B, cols 3-4 of C)
    # Strassen for A[0:2, 0:2] * B[0:2, 3:5]
    # B indices for cols 3,4 are j*5+3, j*5+4
    # b03, b04, b13, b14
    # M1 = (a00+a11)(b03+b14)
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1; V[4][col] = 1
    W[9][col] = 1; W[10][col] = 1; W[12][col] = 1; W[13][col] = 1
    col += 1
    # M2 = (a00+a11)(b13-b03)
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1; V[9][col] = -1
    W[9][col] = -1; W[10][col] = -1; W[12][col] = 1; W[13][col] = 1
    col += 1
    # M3 = a00(b13-b14)
    U[0][col] = 1
    V[8][col] = 1; V[9][col] = -1
    W[9][col] = -1; W[10][col] = -1
    col += 1
    # M4 = a11(b03-b04)
    U[1][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[12][col] = 1; W[13][col] = 1
    col += 1
    # M5 = (a00+a01)b14
    U[0][col] = 1; U[2][col] = 1
    V[4][col] = 1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    # M6 = (a10-a00)(b03+b04)
    U[3][col] = 1; U[0][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[12][col] = -1; W[13][col] = -1
    col += 1
    # M7 = (a01-a11)(b13+b14)
    U[2][col] = 1; U[1][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # Cross terms for 3x3x2
    # A12 B21: a02 b23, a02 b24, a12 b23, a12 b24
    add_term(2, 13, 9, 1, 1, 1, col); col += 1 # a02 b23 -> c03
    add_term(2, 14, 10, 1, 1, 1, col); col += 1 # a02 b24 -> c04
    add_term(5, 13, 12, 1, 1, 1, col); col += 1 # a12 b23 -> c13
    add_term(5, 14, 13, 1, 1, 1, col); col += 1 # a12 b24 -> c14
    
    # A21 B11: a20 b03, a20 b13, a21 b03, a21 b13
    add_term(6, 3, 15, 1, 1, 1, col); col += 1 # a20 b03 -> c23
    add_term(6, 8, 16, 1, 1, 1, col); col += 1 # a20 b13 -> c24
    add_term(7, 3, 15, 1, 1, 1, col); col += 1 # a21 b03 -> c23
    add_term(7, 8, 16, 1, 1, 1, col); col += 1 # a21 b13 -> c24
    
    # A22 B21: a22 b23, a22 b24
    add_term(8, 13, 15, 1, 1, 1, col); col += 1 # a22 b23 -> c23
    add_term(8, 14, 16, 1, 1, 1, col); col += 1 # a22 b24 -> c24
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
