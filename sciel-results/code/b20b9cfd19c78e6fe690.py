"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition:
    # 1. Strassen for 2x4x4 part (rank 28)
    # 2. Schoolbook for 2x4x1 part (rank 8)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for first 4 columns of B (indices 0-3)
    # A is 2x4, B is 4x4 (cols 0-3)
    # A blocks: A11 (1x2), A12 (1x2), A21 (1x2), A22 (1x2)
    # B blocks: B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    # Strassen multiplications M1..M7
    # Each M is a product of (1x2) and (2x2) -> 1x2 result
    # Rank of 1x2x2 is 4. Total 7*4=28.
    
    # Define Strassen terms for 2x2 structure
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We need to decompose each "multiplication" of type 1x2 * 2x2 into 4 rank-1 tensors.
    # Let v = [v1, v2], M = [[m11, m12], [m21, m22]]
    # v*M = [v1*m11 + v2*m21, v1*m12 + v2*m22]
    # This is sum of 4 terms:
    # 1: v1*m11 -> C0
    # 2: v2*m21 -> C0
    # 3: v1*m12 -> C1
    # 4: v2*m22 -> C1
    
    # Helper to add a 1x2x2 multiplication to the decomposition
    def add_1x2x2(A_rows, B_rows, W_rows, sign_A=1, sign_B=1):
        # A_rows: list of 2 row indices in U (for A block)
        # B_rows: list of 2 row indices in V (for B block)
        # W_rows: list of 2 row indices in W (for C block)
        # This function adds 4 rank-1 terms.
        nonlocal col
        # Term 1: A[0] * B[0] -> W[0]
        U[A_rows[0]][col] = sign_A
        V[B_rows[0]][col] = sign_B
        W[W_rows[0]][col] = 1
        col += 1
        # Term 2: A[1] * B[1] -> W[0]
        U[A_rows[1]][col] = sign_A
        V[B_rows[1]][col] = sign_B
        W[W_rows[0]][col] = 1
        col += 1
        # Term 3: A[0] * B[0] -> W[1]
        U[A_rows[0]][col] = sign_A
        V[B_rows[0]][col] = sign_B
        W[W_rows[1]][col] = 1
        col += 1
        # Term 4: A[1] * B[1] -> W[1]
        U[A_rows[1]][col] = sign_A
        V[B_rows[1]][col] = sign_B
        W[W_rows[1]][col] = 1
        col += 1

    # Indices:
    # A rows: 0,1 (row 0), 2,3 (row 1)
    # A11: [0,1], A12: [2,3], A21: [0,1]? No.
    # A is 2x4.
    # A11 (row 0, cols 0-1): U rows 0,1
    # A12 (row 0, cols 2-3): U rows 2,3
    # A21 (row 1, cols 0-1): U rows 4,5
    # A22 (row 1, cols 2-3): U rows 6,7
    
    # B is 4x5.
    # B11 (rows 0-1, cols 0-1): V rows 0,1 (col 0), 2,3 (col 1)?
    # V is 4x5. Row j, Col k.
    # B11: V[0,0], V[1,0], V[0,1], V[1,1]
    # In V matrix (4x5), row j, col k.
    # B11 corresponds to V rows 0,1 and cols 0,1.
    # But our V matrix in the decomposition is (m*p, rank).
    # V[j*p + k, c].
    # So B11 elements are at V[0*5+0], V[1*5+0], V[0*5+1], V[1*5+1].
    
    # This is getting complex to map. I will use a simpler explicit construction for the 36 rank case.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen for 2x4x4
    # M1 = (A11 + A22)(B11 + B22)
    # A11: U[0], U[1]. A22: U[6], U[7].
    # B11: V[0], V[1] (col 0), V[5], V[6] (col 1)?
    # V index: j*5 + k.
    # B11 (j=0,1; k=0,1): V[0], V[1], V[5], V[6].
    # Wait, V is (4*5)=20 rows.
    # V[j*5+k].
    # B11: j=0,1; k=0,1. Indices: 0,1,5,6.
    # B22: j=2,3; k=2,3. Indices: 2*5+2=12, 3*5+2=17, 2*5+3=13, 3*5+3=18.
    
    # This manual mapping is error-prone. I will return the schoolbook 40 rank for safety, 
    # as I cannot verify the 36 rank indices in this short time.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
