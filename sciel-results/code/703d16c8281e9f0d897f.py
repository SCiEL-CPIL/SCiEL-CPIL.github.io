"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: C_0 (Row 0) = A_0 (1x3) * B (3x7)
    # C_0[k] = sum_{j=0}^2 A[0,j] B[j,k]
    for j in range(3):
        for k in range(7):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2a: C_2a (Rows 1,2) = A_2a (2x1) * B_a (1x7)
    # A_2a is col 0 of A rows 1,2. B_a is row 0 of B.
    # C_i[k] = A[i,0] B[0,k] for i in {1,2}
    for i in range(1, 3):
        for k in range(7):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2b: C_2b (Rows 1,2) = A_2b (2x2) * B_b (2x7)
    # A_2b is cols 1,2 of A rows 1,2.
    # B_b is rows 1,2 of B.
    # Split B_b cols into 0,1; 2,3; 4,5; 6.
    # Strassen for 2x2 blocks.
    
    # Helper to add a Strassen block
    def add_strassen(a_i, a_j, b_i, b_j, b_k, b_l, c_i, c_j, offset):
        # A block indices: rows a_i, a_j+1? No, A_2b is fixed.
        # A_2b is rows 1,2 of A, cols 1,2 of A.
        # Let's define A_2b local indices:
        # A11 = A[1,1], A12 = A[1,2]
        # A21 = A[2,1], A22 = A[2,2]
        # B block indices: rows b_i, b_i+1? No, B_b is rows 1,2 of B.
        # B11 = B[b_i, b_k], B12 = B[b_i, b_l]
        # B21 = B[b_i+1, b_k], B22 = B[b_i+1, b_l]
        # C block indices: rows c_i, c_i+1? C_2b is rows 1,2 of C.
        # C11 = C[c_i, b_k], C12 = C[c_i, b_l]
        # C21 = C[c_i+1, b_k], C22 = C[c_i+1, b_l]
        
        nonlocal col
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[a_i * m + 1][col] = 1
        U[(a_i+1) * m + 2][col] = 1
        V[b_i * p + b_k][col] = 1
        V[(b_i+1) * p + (b_l)][col] = 1
        W[b_k * n + c_i][col] = 1
        W[b_l * n + (c_i+1)][col] = 1
        W[b_k * n + (c_i+1)][col] = -1
        W[b_l * n + c_i][col] = -1
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(a_i+1) * m + 1][col] = 1
        U[(a_i+1) * m + 2][col] = 1
        V[b_i * p + b_k][col] = 1
        W[b_k * n + (c_i+1)][col] = 1
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[a_i * m + 1][col] = 1
        V[b_i * p + b_l][col] = 1
        V[(b_i+1) * p + b_l][col] = -1
        W[b_l * n + c_i][col] = 1
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(a_i+1) * m + 2][col] = 1
        V[(b_i+1) * p + b_k][col] = 1
        V[b_i * p + b_k][col] = -1
        W[b_k * n + (c_i+1)][col] = 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[a_i * m + 1][col] = 1
        U[a_i * m + 2][col] = 1
        V[(b_i+1) * p + b_l][col] = 1
        W[b_l * n + c_i][col] = 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(a_i+1) * m + 1][col] = 1
        U[a_i * m + 1][col] = -1
        V[b_i * p + b_k][col] = 1
        V[b_i * p + b_l][col] = 1
        W[b_k * n + (c_i+1)][col] = 1
        W[b_l * n + (c_i+1)][col] = 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[a_i * m + 2][col] = 1
        U[(a_i+1) * m + 2][col] = -1
        V[(b_i+1) * p + b_k][col] = 1
        V[(b_i+1) * p + b_l][col] = 1
        W[b_k * n + c_i][col] = -1
        W[b_l * n + c_i][col] = -1
        col += 1

    # Apply Strassen for B_b cols 0,1 (global B cols 0,1? No, B_b is rows 1,2. Cols are global 0..6)
    # B_b col 0,1 correspond to global B col 0,1.
    add_strassen(1, 2, 1, 0, 0, 1, 1, 2, 0)
    
    # B_b cols 2,3
    add_strassen(1, 2, 1, 2, 2, 3, 1, 2, 0)
    
    # B_b cols 4,5
    add_strassen(1, 2, 1, 4, 4, 5, 1, 2, 0)
    
    # B_b col 6 (2x1 multiplication)
    # C_i[6] = A[1,1]B[1,6] + A[1,2]B[2,6] (for i=1)
    # C_i[6] = A[2,1]B[1,6] + A[2,2]B[2,6] (for i=2)
    # 4 terms.
    
    # Term 1: A[1,1] B[1,6] -> C[1,6]
    U[1 * m + 1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    # Term 2: A[1,2] B[2,6] -> C[1,6]
    U[1 * m + 2][col] = 1
    V[2 * p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    # Term 3: A[2,1] B[1,6] -> C[2,6]
    U[2 * m + 1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 2][col] = 1
    col += 1
    
    # Term 4: A[2,2] B[2,6] -> C[2,6]
    U[2 * m + 2][col] = 1
    V[2 * p + 6][col] = 1
    W[6 * n + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
