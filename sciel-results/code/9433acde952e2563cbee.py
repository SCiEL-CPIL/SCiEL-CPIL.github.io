"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: a_right * b_bottom (j=2)
    # A is 2x3, B is 3x5.
    # a_right is A[:, 2] (2x1). b_bottom is B[2, :] (1x5).
    # C contribution is C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_left * B_top
    # A_left is A[:, 0:2] (2x2). B_top is B[0:2, :] (2x5).
    # We split B_top columns into:
    # Block 1: k in [0, 1] (2x2) -> Strassen
    # Block 2: k in [2, 3] (2x2) -> Strassen
    # Block 3: k in [4] (2x1) -> Schoolbook
    
    def add_strassen_block(k_offset):
        nonlocal col
        # Strassen for 2x2 * 2x2
        # A indices: (0,0), (0,1), (1,0), (1,1) in local A_left coords
        # B indices: (0,0), (0,1), (1,0), (1,1) in local B_block coords
        # Global A index: i * m + j
        # Global B index: j * p + (k_offset + k_local)
        # Global W index: (k_offset + k_local) * n + i
        
        # P1 = (A00 + A11) * (B00 + B11)
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + (k_offset + 0)][col] += 1
        V[1 * p + (k_offset + 1)][col] += 1
        W[(k_offset + 0) * n + 0][col] += 1
        W[(k_offset + 1) * n + 1][col] += 1
        col += 1
        
        # P2 = (A10 + A11) * B00
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + (k_offset + 0)][col] += 1
        W[(k_offset + 0) * n + 1][col] += 1 # C10
        W[(k_offset + 1) * n + 1][col] -= 1 # C11
        col += 1
        
        # P3 = A00 * (B10 - B11)
        U[0 * m + 0][col] += 1
        V[1 * p + (k_offset + 0)][col] += 1
        V[1 * p + (k_offset + 1)][col] -= 1
        W[(k_offset + 0) * n + 0][col] += 1 # C01
        W[(k_offset + 1) * n + 1][col] += 1 # C11
        col += 1
        
        # P4 = A11 * (B01 - B00)
        U[1 * m + 1][col] += 1
        V[0 * p + (k_offset + 1)][col] += 1
        V[0 * p + (k_offset + 0)][col] -= 1
        W[(k_offset + 0) * n + 0][col] += 1 # C00
        W[(k_offset + 0) * n + 1][col] += 1 # C10
        col += 1
        
        # P5 = (A00 + A10) * B11
        U[0 * m + 0][col] += 1
        U[1 * m + 0][col] += 1
        V[1 * p + (k_offset + 1)][col] += 1
        W[(k_offset + 0) * n + 0][col] -= 1 # C00
        W[(k_offset + 0) * n + 1][col] += 1 # C01
        col += 1
        
        # P6 = (A11 - A00) * (B00 + B01)
        U[1 * m + 1][col] += 1
        U[0 * m + 0][col] -= 1
        V[0 * p + (k_offset + 0)][col] += 1
        V[0 * p + (k_offset + 1)][col] += 1
        W[(k_offset + 1) * n + 1][col] += 1 # C11
        col += 1
        
        # P7 = (A01 - A11) * (B10 + B11)
        U[0 * m + 1][col] += 1
        U[1 * m + 1][col] -= 1
        V[1 * p + (k_offset + 0)][col] += 1
        V[1 * p + (k_offset + 1)][col] += 1
        W[(k_offset + 0) * n + 0][col] += 1 # C00
        W[(k_offset + 0) * n + 1][col] += 1 # C01
        col += 1

    # Block 1: k in 0, 1
    add_strassen_block(0)
    
    # Block 2: k in 2, 3
    add_strassen_block(2)
    
    # Block 3: k in 4 (Schoolbook 2x2 * 2x1)
    # C[i, 4] = A[i, 0] * B[0, 4] + A[i, 1] * B[1, 4]
    for i in range(n):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
