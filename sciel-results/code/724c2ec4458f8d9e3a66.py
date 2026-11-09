"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-34 decomposition: Strassen for 2x2x2 sub-blocks where possible.
    # A: 3x3, B: 3x4, C: 3x4.
    # C = A * B.
    # Split A into A_top (2x3) and A_bottom (1x3).
    # Split B into B_left (3x2) and B_right (3x2).
    #
    # C_top_left (2x2) = A_top[:, 0:2] * B_left[0:2, :]  (2x2 * 2x2) -> Strassen rank 7
    # C_top_right (2x2) = A_top[:, 0:2] * B_left[2, :] * B_right[0:2, :] ... wait, dimensions.
    # A is 3x3, B is 3x4.
    # C[i, k] = sum_j A[i, j] B[j, k].
    #
    # Let's partition indices:
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}.
    #
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}. (2x2x2) -> Strassen (7)
    # Block 2: i in {0,1}, j in {2}, k in {0,1}. (2x1x2) -> Schoolbook (4)
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}. (2x2x2) -> Strassen (7)
    # Block 4: i in {0,1}, j in {2}, k in {2,3}. (2x1x2) -> Schoolbook (4)
    # Block 5: i in {2}, j in {0,1,2}, k in {0,1,2,3}. (1x3x4) -> Schoolbook (12)
    #
    # Total rank: 7 + 4 + 7 + 4 + 12 = 34.

    n, m, p = DIMENSIONS  # 3, 3, 4
    rank = 34
    
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Helper to add Strassen decomposition for a 2x2x2 block
    # A indices: (i0, i1) x (j0, j1)
    # B indices: (j0, j1) x (k0, k1)
    # C indices: (i0, i1) x (k0, k1)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # Strassen formulas:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W vectors.
        # U row index: i*m + j
        # V row index: j*p + k
        # W row index: k*n + i
        
        # M1: (A00 + A11) * (B00 + B11)
        # U: A00 at (i0, j0), A11 at (i1, j1)
        # V: B00 at (j0, k0), B11 at (j1, k1)
        # W: C00 at (k0, i0), C11 at (k1, i1)
        # Contribution to C00: +1, to C11: +1
        U[i0*m + j0][col] = 1
        U[i1*m + j1][col] = 1
        V[j0*p + k0][col] = 1
        V[j1*p + k1][col] = 1
        W[k0*n + i0][col] = 1
        W[k1*n + i1][col] = 1
        col += 1
        
        # M2: (A10 + A11) * B00
        # U: A10 at (i1, j0), A11 at (i1, j1)
        # V: B00 at (j0, k0)
        # W: C10 at (k0, i1)
        U[i1*m + j0][col] = 1
        U[i1*m + j1][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i1][col] = 1
        col += 1
        
        # M3: A00 * (B01 - B11)
        # U: A00 at (i0, j0)
        # V: B01 at (j0, k1), B11 at (j1, k1) with -1
        # W: C01 at (k1, i0)
        U[i0*m + j0][col] = 1
        V[j0*p + k1][col] = 1
        V[j1*p + k1][col] = -1
        W[k1*n + i0][col] = 1
        col += 1
        
        # M4: A11 * (B10 - B00)
        # U: A11 at (i1, j1)
        # V: B10 at (j1, k0), B00 at (j0, k0) with -1
        # W: C00 at (k0, i0) with -1, C10 at (k0, i1) with +1
        U[i1*m + j1][col] = 1
        V[j1*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = -1
        W[k0*n + i1][col] = 1
        col += 1
        
        # M5: (A00 + A01) * B11
        # U: A00 at (i0, j0), A01 at (i0, j1)
        # V: B11 at (j1, k1)
        # W: C00 at (k0, i0) with -1, C01 at (k1, i0) with +1
        U[i0*m + j0][col] = 1
        U[i0*m + j1][col] = 1
        V[j1*p + k1][col] = 1
        W[k0*n + i0][col] = -1
        W[k1*n + i0][col] = 1
        col += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        # U: A10 at (i1, j0) with +1, A00 at (i0, j0) with -1
        # V: B00 at (j0, k0), B01 at (j0, k1)
        # W: C11 at (k1, i1)
        U[i1*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + k1][col] = 1
        W[k1*n + i1][col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        # U: A01 at (i0, j1) with +1, A11 at (i1, j1) with -1
        # V: B10 at (j1, k0), B11 at (j1, k1)
        # W: C00 at (k0, i0) with +1, C11 at (k1, i1) with -1
        U[i0*m + j1][col] = 1
        U[i1*m + j1][col] = -1
        V[j1*p + k0][col] = 1
        V[j1*p + k1][col] = 1
        W[k0*n + i0][col] = 1
        W[k1*n + i1][col] = -1
        col += 1
        
    # Helper to add schoolbook decomposition for a block
    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: i in {0,1}, j in {2}, k in {0,1}
    add_schoolbook([0, 1], [2], [0, 1])
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 4: i in {0,1}, j in {2}, k in {2,3}
    add_schoolbook([0, 1], [2], [2, 3])
    
    # Block 5: i in {2}, j in {0,1,2}, k in {0,1,2,3}
    add_schoolbook([2], [0, 1, 2], [0, 1, 2, 3])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
