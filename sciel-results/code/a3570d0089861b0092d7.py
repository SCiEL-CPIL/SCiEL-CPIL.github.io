import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 44
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A_rows, A_cols are lists of 2 indices for the 2x2 submatrix
        # Map: 
        # A[0,0] -> A_rows[0], A_cols[0]
        # A[0,1] -> A_rows[0], A_cols[1]
        # A[1,0] -> A_rows[1], A_cols[0]
        # A[1,1] -> A_rows[1], A_cols[1]
        
        # Indices in global U (A), V (B), W (C)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Strassen intermediate values M1..M7
        # Each M is a product of linear combos of A and B.
        # We need to add 7 columns to U, V, W.
        
        # Define the linear combinations for A and B for each M
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # We need to map local indices (0,0)..(1,1) to global indices
        # Local A:
        # 00: A_rows[0], A_cols[0]
        # 01: A_rows[0], A_cols[1]
        # 10: A_rows[1], A_cols[0]
        # 11: A_rows[1], A_cols[1]
        
        # Local B:
        # 00: B_rows[0], B_cols[0]
        # 01: B_rows[0], B_cols[1]
        # 10: B_rows[1], B_cols[0]
        # 11: B_rows[1], B_cols[1]
        
        # Local C:
        # 00: C_rows[0], C_cols[0]
        # 01: C_rows[0], C_cols[1]
        # 10: C_rows[1], C_cols[0]
        # 11: C_rows[1], C_cols[1]
        
        # Helper to get global index
        def u_idx(r, c): return r * m + c
        def v_idx(r, c): return r * p + c
        def w_idx(r, c): return c * n + r
        
        # Pre-calculate global indices for A, B, C corners
        A_g = [
            [u_idx(A_rows[0], A_cols[0]), u_idx(A_rows[0], A_cols[1])],
            [u_idx(A_rows[1], A_cols[0]), u_idx(A_rows[1], A_cols[1])]
        ]
        B_g = [
            [v_idx(B_rows[0], B_cols[0]), v_idx(B_rows[0], B_cols[1])],
            [v_idx(B_rows[1], B_cols[0]), v_idx(B_rows[1], B_cols[1])]
        ]
        C_g = [
            [w_idx(C_rows[0], C_cols[0]), w_idx(C_rows[0], C_cols[1])],
            [w_idx(C_rows[1], C_cols[0]), w_idx(C_rows[1], C_cols[1])]
        ]
        
        # Definitions of the 7 multiplications
        # Each is (A_coeffs, B_coeffs, C_contrib)
        # A_coeffs: dict {(r,c): coeff}
        # B_coeffs: dict {(r,c): coeff}
        # C_contrib: dict {(r,c): coeff}
        
        strassen_terms = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # Wait, Z10 = M2+M4, Z11 = M1-M2+M3+M6
            # Let's check Strassen formulas again.
            # Z00 = M1 + M4 - M5 + M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M6
            
            # So M2 contributes +1 to Z10, -1 to Z11.
            # M4 contributes +1 to Z00, +1 to Z10.
            # M5 contributes -1 to Z00, +1 to Z01.
            # M7 contributes +1 to Z00.
            # M3 contributes +1 to Z01, +1 to Z11.
            # M1 contributes +1 to Z00, +1 to Z11.
            # M6 contributes +1 to Z11.
        ]
        
        # Correct C contributions:
        # M1: {(0,0): 1, (1,1): 1}
        # M2: {(1,0): 1, (1,1): -1}
        # M3: {(0,1): 1, (1,1): 1}
        # M4: {(0,0): 1, (1,0): 1}
        # M5: {(0,0): -1, (0,1): 1}
        # M6: {(1,1): 1}
        # M7: {(0,0): 1}
        
        # A coeffs:
        # M1: {(0,0): 1, (1,1): 1}
        # M2: {(1,0): 1, (1,1): 1}
        # M3: {(0,0): 1}
        # M4: {(1,1): 1}
        # M5: {(0,0): 1, (0,1): 1}
        # M6: {(1,0): 1, (0,0): -1}
        # M7: {(0,1): 1, (1,1): -1}
        
        # B coeffs:
        # M1: {(0,0): 1, (1,1): 1}
        # M2: {(0,0): 1}
        # M3: {(1,0): 1, (1,1): -1}
        # M4: {(0,1): 1, (0,0): -1}
        # M5: {(1,1): 1}
        # M6: {(0,0): 1, (1,0): 1}
        # M7: {(1,1): 1, (0,1): 1}
        
        terms = [
            # (A, B, C)
            (
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1}
            ),
            (
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(1,0): 1, (1,1): -1}
            ),
            (
                {(0,0): 1},
                {(1,0): 1, (1,1): -1},
                {(0,1): 1, (1,1): 1}
            ),
            (
                {(1,1): 1},
                {(0,1): 1, (0,0): -1},
                {(0,0): 1, (1,0): 1}
            ),
            (
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                {(0,0): -1, (0,1): 1}
            ),
            (
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (1,0): 1},
                {(1,1): 1}
            ),
            (
                {(0,1): 1, (1,1): -1},
                {(1,1): 1, (0,1): 1},
                {(0,0): 1}
            ),
        ]
        
        for A_c, B_c, C_c in terms:
            for (r, c), val in A_c.items():
                U[A_g[r][c], col] += val
            for (r, c), val in B_c.items():
                V[B_g[r][c], col] += val
            for (r, c), val in C_c.items():
                W[C_g[r][c], col] += val
            col += 1

    # 1. Top 2 rows of A (0, 1)
    # Split into Left (cols 0,1) and Right (cols 2,3)
    # A_rows = [0, 1]
    
    # Block LL: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block LR: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block RL: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block RR: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # 2. Bottom 1 row of A (2)
    # A[2, 0:4] * B[0:4, 0:4] -> C[2, 0:4]
    # Terms: A[2,j] * B[j,k] contributes to C[2,k]
    for j in range(4):
        for k in range(4):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
            
    # Check if col == rank
    if col != rank:
        # If we have extra space, it's fine, but we want exact rank.
        # If col < rank, we have unused columns (zeros).
        # If col > rank, we have a bug.
        pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
