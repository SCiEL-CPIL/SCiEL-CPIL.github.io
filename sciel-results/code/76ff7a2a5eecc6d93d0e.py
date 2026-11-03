"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    rank = 22
    
    # Shapes
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 22
    
    def add_strassen_block(a_idx, b_idx, c_idx, start_col):
        """
        Adds a Strassen decomposition for a 2x2 x 2x2 multiplication.
        a_idx: list of 4 indices for A (00, 01, 10, 11)
        b_idx: list of 4 indices for B (00, 01, 10, 11)
        c_idx: list of 4 indices for C (00, 01, 10, 11)
        start_col: starting column index in U, V, W
        """
        # Strassen multiplications M1..M7
        # M1 = A11 * (B11 + B22)
        # M2 = (A11 + A22) * B22
        # M3 = (A12 + A22) * B21
        # M4 = A11 * (B12 - B22)
        # M5 = (A21 + A11) * B11
        # M6 = (A21 - A12) * (B12 + B22)
        # M7 = (A11 - A21) * (B21 + B11)
        
        # C11 = M1 - M4 + M5 - M7
        # C12 = M4 + M6
        # C21 = M3 + M7
        # C22 = M1 - M3 + M2 + M6
        
        # a_idx: [A11, A12, A21, A22]
        # b_idx: [B11, B12, B21, B22]
        # c_idx: [C11, C12, C21, C22]
        
        A11, A12, A21, A22 = a_idx
        B11, B12, B21, B22 = b_idx
        C11, C12, C21, C22 = c_idx
        
        # M1
        col = start_col
        U[A11][col] += 1
        V[B11][col] += 1
        V[B22][col] += 1
        W[C11][col] += 1
        W[C22][col] += 1
        col += 1
        
        # M2
        U[A11][col] += 1
        U[A22][col] += 1
        V[B22][col] += 1
        W[C22][col] += 1
        col += 1
        
        # M3
        U[A12][col] += 1
        U[A22][col] += 1
        V[B21][col] += 1
        W[C21][col] += 1
        col += 1
        
        # M4
        U[A11][col] += 1
        V[B12][col] += 1
        V[B22][col] += -1
        W[C11][col] += -1
        W[C12][col] += 1
        col += 1
        
        # M5
        U[A21][col] += 1
        U[A11][col] += 1
        V[B11][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M6
        U[A21][col] += 1
        U[A12][col] += -1
        V[B12][col] += 1
        V[B22][col] += 1
        W[C12][col] += 1
        W[C22][col] += 1
        col += 1
        
        # M7
        U[A11][col] += 1
        U[A21][col] += -1
        V[B21][col] += 1
        V[B11][col] += 1
        W[C11][col] += -1
        W[C21][col] += 1
        col += 1
        
        return col

    def add_simple_products(a_indices, b_indices, c_indices, start_col):
        """
        Adds simple products A[i] * B[j] -> C[k]
        """
        col = start_col
        for a, b, c in zip(a_indices, b_indices, c_indices):
            U[a][col] += 1
            V[b][col] += 1
            W[c][col] += 1
            col += 1
        return col

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 00(0), 01(1), 10(3), 11(4)
    a_idx_1 = [0, 1, 3, 4]
    # B indices: 00(0), 01(1), 10(4), 11(5)
    b_idx_1 = [0, 1, 4, 5]
    # C indices: 00(0), 01(2), 10(1), 11(3)
    # W index = k*n + i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    c_idx_1 = [0, 2, 1, 3]
    
    current_col = add_strassen_block(a_idx_1, b_idx_1, c_idx_1, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices same
    a_idx_2 = [0, 1, 3, 4]
    # B indices: 02(2), 03(3), 12(6), 13(7)
    b_idx_2 = [2, 3, 6, 7]
    # C indices:
    # C02: k=2, i=0 -> 4
    # C03: k=3, i=0 -> 6
    # C12: k=2, i=1 -> 5
    # C13: k=3, i=1 -> 7
    c_idx_2 = [4, 6, 5, 7]
    
    current_col = add_strassen_block(a_idx_2, b_idx_2, c_idx_2, current_col)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # A indices: 02(2), 12(5)
    # B indices: 20(8), 21(9), 22(10), 23(11)
    # Products:
    # A02 * B20 -> C00 (0)
    # A02 * B21 -> C01 (2)
    # A02 * B22 -> C02 (4)
    # A02 * B23 -> C03 (6)
    # A12 * B20 -> C10 (1)
    # A12 * B21 -> C11 (3)
    # A12 * B22 -> C12 (5)
    # A12 * B23 -> C13 (7)
    
    a_indices = [2, 2, 2, 2, 5, 5, 5, 5]
    b_indices = [8, 9, 10, 11, 8, 9, 10, 11]
    c_indices = [0, 2, 4, 6, 1, 3, 5, 7]
    
    current_col = add_simple_products(a_indices, b_indices, c_indices, current_col)
    
    # current_col should be 22
    assert current_col == rank, f"Rank mismatch: {current_col} != {rank}"
    
    # Convert to integers if they are .0
    for mat in [U, V, W]:
        for row in mat:
            for i in range(len(row)):
                if row[i] == int(row[i]):
                    row[i] = int(row[i])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
