"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x5 matrix using blocking and Strassen's algorithm.
    
    Decomposition:
    A = [A1, A2] where A1, A2 are 2x2.
    B = [[B11, B12, B13],
         [B21, B22, B23]] where Bij (j<3) are 2x2, Bij (j=3) are 2x1.
    C1 = A1*B11 + A2*B21 (2x2)
    C2 = A1*B12 + A2*B22 (2x2)
    C3 = A1*B13 + A2*B23 (2x1)
    
    Use Strassen (rank 7) for 2x2x2 products.
    Use Schoolbook (rank 4) for 2x2x1 products.
    Total rank = 4 * 7 + 2 * 4 = 36.
    """
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(A_idx, B_idx, C_idx):
        """
        Adds Strassen decomposition for 2x2x2 multiplication.
        A_idx: list of 4 indices for A (row-major: 00, 01, 10, 11)
        B_idx: list of 4 indices for B (row-major: 00, 01, 10, 11)
        C_idx: list of 4 indices for C (row-major: 00, 01, 10, 11)
        """
        nonlocal col
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        c00, c01, c10, c11 = C_idx
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] -= 1
        col += 1

    def add_schoolbook_2x2x1(A_idx, B_idx, C_idx):
        """
        Adds schoolbook decomposition for 2x2x1 multiplication.
        A_idx: list of 4 indices for A (row-major)
        B_idx: list of 2 indices for B (row-major)
        C_idx: list of 2 indices for C (row-major)
        """
        nonlocal col
        a00, a01, a10, a11 = A_idx
        b0, b1 = B_idx
        c0, c1 = C_idx
        
        # c0 = a00*b0 + a01*b1
        U[a00][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1
        col += 1
        
        U[a01][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        col += 1
        
        # c1 = a10*b0 + a11*b1
        U[a10][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1
        col += 1
        
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1

    # Define indices
    # A is 2x4: indices 0-7
    # A1: 0,1,4,5
    # A2: 2,3,6,7
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    # B is 4x5: indices 0-19
    # B11 (rows 0-1, cols 0-1): 0,1,5,6
    # B12 (rows 0-1, cols 2-3): 2,3,7,8
    # B13 (rows 0-1, col 4):    4,9
    # B21 (rows 2-3, cols 0-1): 10,11,15,16
    # B22 (rows 2-3, cols 2-3): 12,13,17,18
    # B23 (rows 2-3, col 4):    14,19
    
    B11_idx = [0, 1, 5, 6]
    B12_idx = [2, 3, 7, 8]
    B13_idx = [4, 9]
    B21_idx = [10, 11, 15, 16]
    B22_idx = [12, 13, 17, 18]
    B23_idx = [14, 19]
    
    # C is 2x5: indices 0-9 (k*n+i)
    # C1 (cols 0-1): 0,1,2,3
    # C2 (cols 2-3): 4,5,6,7
    # C3 (col 4):    8,9
    
    C1_idx = [0, 1, 2, 3]
    C2_idx = [4, 5, 6, 7]
    C3_idx = [8, 9]
    
    # C1 = A1*B11 + A2*B21
    add_strassen(A1_idx, B11_idx, C1_idx)
    add_strassen(A2_idx, B21_idx, C1_idx)
    
    # C2 = A1*B12 + A2*B22
    add_strassen(A1_idx, B12_idx, C2_idx)
    add_strassen(A2_idx, B22_idx, C2_idx)
    
    # C3 = A1*B13 + A2*B23
    add_schoolbook_2x2x1(A1_idx, B13_idx, C3_idx)
    add_schoolbook_2x2x1(A2_idx, B23_idx, C3_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
