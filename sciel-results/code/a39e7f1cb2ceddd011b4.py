DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        # U
        U[i0 * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        # V
        V[j0 * p + k0][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        # W
        W[k0 * n + i0][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1
        col += 1
        
        # M2
        U[i0 * m + j0][col] += 1 # A10? No, A_{10} is i0+1, j0. Wait.
        # Strassen indices:
        # A is 2x2: A00, A01; A10, A11.
        # M2 = (A10 + A11) B00.
        # A10 is i0+1, j0. A11 is i0+1, j0+1.
        # B00 is j0, k0.
        U[(i0+1) * m + j0][col] += 1
        U[(i0+1) * m + (j0+1)][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + (i0+1)][col] += 1 # C10
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i0 * m + j0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] -= 1
        W[(k0+1) * n + i0][col] += 1 # C01
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[(i0+1) * m + (j0+1)][col] += 1
        V[(j0+1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + (i0+1)][col] += 1 # C10
        W[(k0+1) * n + (i0+1)][col] += 1 # C11? No, M4 is in C00 and C10.
        # C00 = M1 + M4 ...
        # C10 = M2 + M4 ...
        # So W for M4: C00 (i0, k0) and C10 (i0+1, k0).
        W[k0 * n + i0][col] += 1
        W[k0 * n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i0 * m + j0][col] += 1
        U[i0 * m + (j0+1)][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        W[(k0+1) * n + i0][col] += 1 # C01
        W[(k0+1) * n + (i0+1)][col] -= 1 # C11? No, C11 = M1 - M2 + M3 + M6. M5 is not in C11.
        # C00 = M1 + M4 - M5 + M7.
        # So M5 contributes to C00 with -1 and C01 with +1.
        W[k0 * n + i0][col] -= 1
        W[(k0+1) * n + i0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[(i0+1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + (k0+1)][col] += 1
        W[(k0+1) * n + (i0+1)][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[i0 * m + (j0+1)][col] += 1
        U[(i0+1) * m + (j0+1)][col] -= 1
        V[(j0+1) * p + k0][col] += 1
        V[(j0+1) * p + (k0+1)][col] += 1
        W[k0 * n + i0][col] += 1 # C00
        col += 1

    # Helper to add Schoolbook 1x4x4 block
    def add_schoolbook_1x4x4(i0, j0, k0):
        nonlocal col
        # i0 is the row index (must be 2)
        # j0 is start col of A / row of B
        # k0 is start col of B / col of C
        for j in range(4):
            for k in range(4):
                U[i0 * m + (j0 + j)][col] += 1
                V[(j0 + j) * p + (k0 + k)][col] += 1
                W[(k0 + k) * n + i0][col] += 1
                col += 1

    # Component 1: A1 BL (2x4 x 4x4)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Component 2: A2 BL (1x4 x 4x4)
    add_schoolbook_1x4x4(2, 0, 0)
    
    # Component 3: A1 BR (2x4 x 4x4)
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_strassen(0, 2, 4)
    add_strassen(0, 2, 6)
    
    # Component 4: A2 BR (1x4 x 4x4)
    add_schoolbook_1x4x4(2, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
