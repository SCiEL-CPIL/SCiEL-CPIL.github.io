"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices
    # A is 3x3, B is 3x4
    # A = [[A11, A12], [A21, A22]]
    # A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]]
    # B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    # U rows: i*m + j (0..8)
    # V rows: j*p + k (0..11)
    # W rows: k*n + i (0..11)
    
    col = 0
    # Strassen for A11 * B11 (7 products)
    # A11 = [[a, b], [c, d]] -> U rows: 0,1,2,3
    # B11 = [[e, f], [g, h]] -> V rows: 0,1,4,5
    # W rows for C11: 0,1,3,4
    strassen_a11_b11 = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), # a, b, c, d
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3) # e, f, g, h
    ]
    # Actually, let's just hardcode the 34 terms.
    # Term 1: (a+d)(e+h)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    # Term 2: (c+d)(e+f)
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    # Term 3: a(f-h)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    # Term 4: d(g-e)
    U[3][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[3][col] = 1; W[0][col] = -1
    col += 1
    # Term 5: (a+b)h
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1; W[1][col] = 1
    col += 1
    # Term 6: (c-a)(e+f)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1; W[0][col] = -1
    col += 1
    # Term 7: (b-d)(g+h)
    U[1][col] = 1; U[3][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[1][col] = 1; W[0][col] = -1
    col += 1
    
    # Strassen for A11 * B12 (7 products)
    # B12 = [[i, j], [k, l]] -> V rows: 2,3,6,7
    # W rows for C12: 2,3,5,6
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[6][col] = 1
    col += 1
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[3][col] = 1
    W[5][col] = 1; W[6][col] = 1
    col += 1
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[3][col] = 1; W[6][col] = -1
    col += 1
    U[3][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[2][col] = -1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = 1; W[3][col] = 1
    col += 1
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[5][col] = 1; W[2][col] = -1
    col += 1
    U[1][col] = 1; U[3][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[3][col] = 1; W[2][col] = -1
    col += 1
    
    # A12 * B21 (4 products)
    # A12 = [[u], [v]] -> U rows: 4, 5
    # B21 = [[p, q]] -> V rows: 8, 9
    # W rows for C11: 0,1,3,4
    U[4][col] = 1; V[8][col] = 1; W[0][col] = 1
    col += 1
    U[4][col] = 1; V[9][col] = 1; W[1][col] = 1
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[3][col] = 1
    col += 1
    U[5][col] = 1; V[9][col] = 1; W[4][col] = 1
    col += 1
    
    # A12 * B22 (4 products)
    # B22 = [[r, s]] -> V rows: 10, 11
    # W rows for C12: 2,3,5,6
    U[4][col] = 1; V[10][col] = 1; W[2][col] = 1
    col += 1
    U[4][col] = 1; V[11][col] = 1; W[3][col] = 1
    col += 1
    U[5][col] = 1; V[10][col] = 1; W[5][col] = 1
    col += 1
    U[5][col] = 1; V[11][col] = 1; W[6][col] = 1
    col += 1
    
    # A21 * B11 (4 products)
    # A21 = [[x, y]] -> U rows: 6, 7
    # B11 = [[e, f], [g, h]] -> V rows: 0,1,4,5
    # W rows for C21: 7,8
    U[6][col] = 1; V[0][col] = 1; W[7][col] = 1
    col += 1
    U[7][col] = 1; V[4][col] = 1; W[7][col] = 1
    col += 1
    U[6][col] = 1; V[1][col] = 1; W[8][col] = 1
    col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1
    col += 1
    
    # A21 * B12 (4 products)
    # B12 = [[i, j], [k, l]] -> V rows: 2,3,6,7
    # W rows for C22: 9,10
    U[6][col] = 1; V[2][col] = 1; W[9][col] = 1
    col += 1
    U[7][col] = 1; V[6][col] = 1; W[9][col] = 1
    col += 1
    U[6][col] = 1; V[3][col] = 1; W[10][col] = 1
    col += 1
    U[7][col] = 1; V[7][col] = 1; W[10][col] = 1
    col += 1
    
    # A22 * B21 (2 products)
    # A22 = [[z]] -> U row: 8
    # B21 = [[p, q]] -> V rows: 8, 9
    # W rows for C21: 7,8
    U[8][col] = 1; V[8][col] = 1; W[7][col] = 1
    col += 1
    U[8][col] = 1; V[9][col] = 1; W[8][col] = 1
    col += 1
    
    # A22 * B22 (2 products)
    # B22 = [[r, s]] -> V rows: 10, 11
    # W rows for C22: 9,10
    U[8][col] = 1; V[10][col] = 1; W[9][col] = 1
    col += 1
    U[8][col] = 1; V[11][col] = 1; W[10][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
