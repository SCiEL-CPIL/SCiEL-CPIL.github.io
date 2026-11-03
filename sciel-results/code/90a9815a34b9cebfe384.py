"""Rewritten program for rank-17 matrix multiplication 2x3x3."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (9, 17)
    # W: (p*n, rank) = (6, 17)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx: list of (index, val) for U column
        # v_idx: list of (index, val) for V column
        # w_idx: list of (index, val) for W column
        pass # This helper is conceptual, we will fill directly
    
    c = 0
    
    # Strassen Core (7 terms)
    # M1 = (A00 + A11)(B00 + B11)
    U[0][c] = 1; U[4][c] = 1
    V[0][c] = 1; V[4][c] = 1
    W[0][c] = 1; W[3][c] = 1
    c += 1
    
    # M2 = (A10 + A11)B00
    U[3][c] = 1; U[4][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[3][c] = -1
    c += 1
    
    # M3 = A00(B01 - B11)
    U[0][c] = 1
    V[1][c] = 1; V[4][c] = -1
    W[2][c] = 1; W[3][c] = 1
    c += 1
    
    # M4 = A11(B10 - B00)
    U[4][c] = 1
    V[3][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1
    c += 1
    
    # M5 = (A00 + A10)B11
    U[0][c] = 1; U[3][c] = 1
    V[4][c] = 1
    W[0][c] = -1; W[2][c] = 1
    c += 1
    
    # M6 = (A01 - A00)(B00 + B10)
    U[1][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[3][c] = 1
    W[3][c] = 1
    c += 1
    
    # M7 = (A11 - A10)(B01 + B11)
    U[4][c] = 1; U[3][c] = -1
    V[1][c] = 1; V[4][c] = 1
    W[0][c] = 1
    c += 1
    
    # Remainder k in {0,1}: A_i2 * B_2k
    # i=0, k=0
    U[2][c] = 1
    V[6][c] = 1
    W[0][c] = 1
    c += 1
    
    # i=0, k=1
    U[2][c] = 1
    V[7][c] = 1
    W[2][c] = 1
    c += 1
    
    # i=1, k=0
    U[5][c] = 1
    V[6][c] = 1
    W[1][c] = 1
    c += 1
    
    # i=1, k=1
    U[5][c] = 1
    V[7][c] = 1
    W[3][c] = 1
    c += 1
    
    # Remainder k=2: A_i0*B_02 + A_i1*B_12 + A_i2*B_22
    # i=0, j=0
    U[0][c] = 1
    V[2][c] = 1
    W[4][c] = 1
    c += 1
    
    # i=0, j=1
    U[1][c] = 1
    V[5][c] = 1
    W[4][c] = 1
    c += 1
    
    # i=0, j=2
    U[2][c] = 1
    V[8][c] = 1
    W[4][c] = 1
    c += 1
    
    # i=1, j=0
    U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1
    c += 1
    
    # i=1, j=1
    U[4][c] = 1
    V[5][c] = 1
    W[5][c] = 1
    c += 1
    
    # i=1, j=2
    U[5][c] = 1
    V[8][c] = 1
    W[5][c] = 1
    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
