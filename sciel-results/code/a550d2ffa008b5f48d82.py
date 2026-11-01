"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of B.
    # p=16, so we have 8 groups of 2 columns.
    num_groups = p // 2
    rank = 7 * num_groups
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A
    # M1: (A11 + A22)
    # M2: (A21 + A22)
    # M3: A11
    # M4: A22
    # M5: (A11 + A12)
    # M6: (A21 - A11)
    # M7: (A12 - A22)
    
    # Indices for A: 0:11, 1:12, 2:21, 3:22
    a_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # For each group g
    for g in range(num_groups):
        col1 = 2 * g
        col2 = 2 * g + 1
        
        # B coefficients for each M
        # M1: B11 + B22  -> B[0, col1] + B[1, col2]
        # M2: B11        -> B[0, col1]
        # M3: B21 - B11  -> B[1, col1] - B[0, col1]
        # M4: B12 + B22  -> B[0, col2] + B[1, col2]
        # M5: B22        -> B[1, col2]
        # M6: B11 + B12  -> B[0, col1] + B[0, col2]
        # M7: B21 + B22  -> B[1, col1] + B[1, col2]
        
        b_coeffs = [
            [(0, col1, 1), (1, col2, 1)],
            [(0, col1, 1)],
            [(1, col1, 1), (0, col1, -1)],
            [(0, col2, 1), (1, col2, 1)],
            [(1, col2, 1)],
            [(0, col1, 1), (0, col2, 1)],
            [(1, col1, 1), (1, col2, 1)]
        ]
        
        # Output coefficients for C
        # C11 (row 0, col1) = M1 - M4 + M5 + M7
        # C12 (row 0, col2) = M3 + M4
        # C21 (row 1, col1) = M2 + M3 - M1 + M6
        # C22 (row 1, col2) = M1 - M2 + M6 - M7
        
        # W is indexed by [k*n + i, r] where k is col, i is row.
        # So W[col*n + row, r]
        
        for s in range(7):
            r = 7 * g + s
            
            # Fill U
            for idx_a in range(4):
                U[idx_a][r] = a_coeffs[s][idx_a]
                
            # Fill V
            for (row_b, col_b, coeff) in b_coeffs[s]:
                V[row_b * p + col_b][r] = coeff
                
            # Fill W
            # We need to add the contribution of M_s to the output.
            # The output formulas are linear combinations of M1..M7.
            # We can just add the coefficient to W.
            
            if s == 0: # M1
                W[col1 * n + 0][r] += 1  # C11
                W[col2 * n + 0][r] += 0  # C12
                W[col1 * n + 1][r] += -1 # C21
                W[col2 * n + 1][r] += 1  # C22
            elif s == 1: # M2
                W[col1 * n + 0][r] += 0
                W[col2 * n + 0][r] += 0
                W[col1 * n + 1][r] += 1
                W[col2 * n + 1][r] += -1
            elif s == 2: # M3
                W[col1 * n + 0][r] += 0
                W[col2 * n + 0][r] += 1
                W[col1 * n + 1][r] += 1
                W[col2 * n + 1][r] += 0
            elif s == 3: # M4
                W[col1 * n + 0][r] += -1
                W[col2 * n + 0][r] += 1
                W[col1 * n + 1][r] += 0
                W[col2 * n + 1][r] += 0
            elif s == 4: # M5
                W[col1 * n + 0][r] += 1
                W[col2 * n + 0][r] += 0
                W[col1 * n + 1][r] += 0
                W[col2 * n + 1][r] += 0
            elif s == 5: # M6
                W[col1 * n + 0][r] += 0
                W[col2 * n + 0][r] += 0
                W[col1 * n + 1][r] += 1
                W[col2 * n + 1][r] += 1
            elif s == 6: # M7
                W[col1 * n + 0][r] += 1
                W[col2 * n + 0][r] += 0
                W[col1 * n + 1][r] += 0
                W[col2 * n + 1][r] += -1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
