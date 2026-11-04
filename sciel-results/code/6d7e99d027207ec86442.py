"""Matrix multiplication 4x4 x 4x6 via optimized block decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy:
    # Split B into B1 (4x4) and B2 (4x2).
    # C1 = A * B1 (4x4x4) -> Use Recursive Strassen (Rank 49).
    # C2 = A * B2 (4x4x2) -> Use Block-Schoolbook Optimization (Rank 16).
    # Total Rank = 49 + 16 = 65.
    
    # --- Part 1: 4x4x4 using Recursive Strassen (Rank 49) ---
    # Strassen 2x2x2 Rank 7 factors
    # A-side (U):
    # M0: A00+A11, M1: A10+A11, M2: A00, M3: A11, M4: A00+A01, M5: A10-A00, M6: A01-A11
    U_s2 = [
        [1, 0, 1, 0, 1, 0, 0], # A00 (row 0)
        [0, 1, 0, 1, 0, 1, 1], # A01 (row 1) -> Wait, standard Strassen A factors:
        # Let's use standard Strassen A factors for 2x2:
        # M0: A00+A11 -> (0,0):1, (1,1):1
        # M1: A10+A11 -> (1,0):1, (1,1):1
        # M2: A00 -> (0,0):1
        # M3: A11 -> (1,1):1
        # M4: A00+A01 -> (0,0):1, (0,1):1
        # M5: A10-A00 -> (1,0):1, (0,0):-1
        # M6: A01-A11 -> (0,1):1, (1,1):-1
    ]
    # U_s2 is 4x7 (rows 0..3 correspond to A00, A01, A10, A11)
    U_s2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00: M0, M2, M4, M5(-1)
        [0, 0, 0, 0, 1, 0, 1],  # A01: M4, M6
        [0, 1, 0, 0, 0, 1, 0],  # A10: M1, M5
        [1, 1, 0, 1, 0, 0, -1]  # A11: M0, M1, M3, M6(-1)
    ]
    
    # B-side (V):
    # M0: B00+B11 -> (0,0):1, (1,1):1
    # M1: B00 -> (0,0):1
    # M2: B10-B11 -> (1,0):1, (1,1):-1
    # M3: B01-B10 -> (0,1):1, (1,0):-1
    # M4: B11 -> (1,1):1
    # M5: B00+B10 -> (0,0):1, (1,0):1
    # M6: B01+B11 -> (0,1):1, (1,1):1
    # V_s2 is 4x7 (rows 0..3 correspond to B00, B01, B10, B11)
    V_s2 = [
        [1, 1, 0, 0, 0, 1, 0], # B00: M0, M1, M5
        [0, 0, 0, 1, 0, 0, 1], # B01: M3, M6
        [0, 0, 1, -1, 0, 1, 0],# B10: M2, M3(-1), M5
        [1, 0, -1, 0, 1, 0, 1] # B11: M0, M2(-1), M4, M6
    ]
    
    # C-side (W):
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    # W_s2 is 4x7 (rows 0..3 correspond to C00, C01, C10, C11)
    W_s2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Build 4x4x4 factors (Rank 49) via Kronecker Product
    # U_44 = U_s2 (Kron) U_s2 -> 16x49
    # V_44 = V_s2 (Kron) V_s2 -> 16x49
    # W_44 = W_s2 (Kron) W_s2 -> 16x49
    
    def kron(A, B):
        m1, n1 = len(A), len(A[0])
        m2, n2 = len(B), len(B[0])
        C = [[0]*(n1*n2) for _ in range(m1*m2)]
        for i in range(m1):
            for j in range(n1):
                if A[i][j] != 0:
                    for k in range(m2):
                        for l in range(n2):
                            C[i*m2+k][j*n2+l] += A[i][j] * B[k][l]
        return C

    U_44 = kron(U_s2, U_s2)
    V_44 = kron(V_s2, V_s2)
    W_44 = kron(W_s2, W_s2)
    
    # --- Part 2: 4x4x2 using Block-Schoolbook (Rank 16) ---
    # A is 4x4, B2 is 4x2.
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B2 into 2x2 blocks: B00, B10 (top 2 rows, bottom 2 rows).
    # C2 is 4x2. Split into C00, C10 (top 2 rows, bottom 2 rows).
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    # Optimization:
    # X = B00 + B10
    # Y = B00 - B10
    # B00 = (X+Y)/2, B10 = (X-Y)/2
    
    # C00 = 0.5 * (A00+A01)X + 0.5 * (A00-A01)Y
    # C10 = 0.5 * (A10+A11)X + 0.5 * (A10-A11)Y
    
    # We perform 4 matrix multiplications of 2x2 matrices:
    # M1 = (A00+A01) X
    # M2 = (A00-A01) Y
    # M3 = (A10+A11) X
    # M4 = (A10-A11) Y
    
    # Each 2x2 multiplication uses Schoolbook (Rank 4).
    # Total Rank = 4 * 4 = 16.
    
    # U_442: 16x16
    # V_442: 12x16 (4x2 matrix has 8 elements? No, B is 4x2, so 8 elements. Wait.
    # B2 is 4x2.
    # B00 is 2x2 (4 elements). B10 is 2x2 (4 elements).
    # Total B elements = 8.
    # V_442 shape: (m*p_part, rank) = (4*2, 16) = (8, 16).
    # W_442 shape: (p_part*n, rank) = (2*4, 16) = (8, 16).
    
    U_442 = [[0]*16 for _ in range(16)]
    V_442 = [[0]*16 for _ in range(8)]
    W_442 = [[0]*16 for _ in range(8)]
    
    # Schoolbook 2x2x2 factors (Rank 4)
    # A factors: A00, A01, A10, A11
    # B factors: B00, B01, B10, B11
    # C factors: C00, C01, C10, C11
    # Term r: A_ij * B_jk -> C_ik
    
    # We need to map the 4 block-multiplications to the global indices.
    # Block Multiplication 1: (A00+A01) * X
    # Block Multiplication 2: (A00-A01) * Y
    # Block Multiplication 3: (A10+A11) * X
    # Block Multiplication 4: (A10-A11) * Y
    
    # Global A indices:
    # A00: rows 0,1; cols 0,1
    # A01: rows 0,1; cols 2,3
    # A10: rows 2,3; cols 0,1
    # A11: rows 2,3; cols 2,3
    
    # Global B2 indices (4x2):
    # B00: rows 0,1; cols 0,1
    # B10: rows 2,3; cols 0,1
    # Note: B2 is 4x2.
    # Row 0,1 of B2 correspond to B00.
    # Row 2,3 of B2 correspond to B10.
    
    # Global C2 indices (4x2):
    # C00: rows 0,1; cols 0,1
    # C10: rows 2,3; cols 0,1
    
    # Helper to add a schoolbook 2x2x2 multiplication
    # a_rows: list of 4 global A row indices for the 2x2 block
    # a_cols: list of 4 global A col indices for the 2x2 block
    # b_rows: list of 4 global B row indices for the 2x2 block
    # b_cols: list of 4 global B col indices for the 2x2 block
    # c_rows: list of 4 global C row indices for the 2x2 block
    # c_cols: list of 4 global C col indices for the 2x2 block
    # sign_a: 1 or -1 (for A factors)
    # sign_b: 1 or -1 (for B factors, if using X/Y combination)
    # offset: rank offset
    
    def add_schoolbook_2x2(a_r, a_c, b_r, b_c, c_r, c_c, sign_a, sign_b, offset):
        # Schoolbook terms:
        # T0: A00 B00 -> C00
        # T1: A00 B01 -> C01
        # T2: A01 B10 -> C10
        # T3: A01 B11 -> C11
        # T4: A10 B00 -> C10
        # T5: A10 B01 -> C11
        # T6: A11 B10 -> C10
        # T7: A11 B11 -> C11
        
        # Actually, standard schoolbook is 4 multiplications? No, 4x4x4 is 16.
        # 2x2x2 schoolbook is 4 multiplications?
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # This is 8 multiplications.
        # Rank 8 for 2x2x2?
        # Yes, schoolbook rank for 2x2x2 is 8? No, it's 8.
        # Wait, Strassen is 7. Schoolbook is 8.
        
        # So each block multiplication has rank 8.
        # 4 block multiplications * 8 = 32.
        # My previous estimate of 16 was wrong.
        # Rank 32 for 4x4x2.
        
        # Total Rank = 49 + 32 = 81.
        # This is WORSE than 77.
        
        # Let's stick to the Kronecker Product Rank 77 which is known to be correct and simpler to implement correctly.
        pass
    
    # I will revert to the Rank 77 Kronecker Product but fix the implementation.
    
    # --- Correct Rank 77 Implementation ---
    
    # Inner 2x2x3 decomposition (Rank 11)
    # Split 3 cols into 1 (Schoolbook Rank 4) + 2 (Strassen Rank 7).
    
    # Schoolbook 2x2x1 (Rank 4)
    # A 2x2, B 2x1, C 2x1
    # C0 = A00B0 + A01B1
    # C1 = A10B0 + A11B1
    # Terms:
    # T0: A00 B0 -> C0
    # T1: A01 B1 -> C0
    # T2: A10 B0 -> C1
    # T3: A11 B1 -> C1
    
    U_s21 = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    V_s21 = [
        [1, 0, 1, 0], # B0 (row 0 of B)
        [0, 1, 0, 1]  # B1 (row 1 of B)
    ]
    W_s21 = [
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ]
    
    # Strassen 2x2x2 (Rank 7) for cols 1,2
    # We already have U_s2, V_s2, W_s2 for 2x2x2.
    # But V_s2 is 2x2. Here we need 2x2 (cols 1,2).
    # The indices for B in V_s2 are 0,1 corresponding to local cols 0,1.
    # We need to map local col 0 -> global col 1, local col 1 -> global col 2.
    
    # Combine Inner 2x2x3 (Rank 11)
    # U_in: 4x11
    # V_in: 6x11 (2 rows * 3 cols)
    # W_in: 6x11 (3 cols * 2 rows)
    
    U_in = [[0]*11 for _ in range(4)]
    V_in = [[0]*11 for _ in range(6)]
    W_in = [[0]*11 for _ in range(6)]
    
    # Part 1: Schoolbook for col 0 (Rank 4)
    for r in range(4):
        for i in range(4):
            U_in[i][r] = U_s21[i][r]
        for j in range(2):
            V_in[j*3+0][r] = V_s21[j][r]
        for k in range(2):
            W_in[0*2+k][r] = W_s21[k][r]
            
    # Part 2: Strassen for cols 1,2 (Rank 7)
    for r in range(7):
        for i in range(4):
            U_in[i][4+r] = U_s2[i][r]
        for j in range(2):
            for k in range(2):
                # Local col k -> Global col k+1
                V_in[j*3+(k+1)][4+r] = V_s2[j][k] # Wait, V_s2 is 4x7? No, V_s2 is 4x7 for 2x2x2.
                # V_s2 rows are B00, B01, B10, B11.
                # In 2x2x2, B is 2x2.
                # Here B is 2x2 (local).
                # So V_s2[j*2+k] is the coefficient for B_jk.
                # But V_in row index is j*3 + k_global.
                # j is 0,1. k_global is 1,2.
                # V_s2 index: row j, col k.
                # So V_in[j*3+k+1][4+r] = V_s2[j][k] is WRONG.
                # V_s2 is defined as 4x7 matrix where row 0 is B00, row 1 is B01, etc.
                # So V_s2[0] is B00, V_s2[1] is B01, V_s2[2] is B10, V_s2[3] is B11.
                # We need to fill V_in rows for B.
                # B row 0 (j=0): B00, B01.
                # B row 1 (j=1): B10, B11.
                # Global col 1 (local 0): B00, B10.
                # Global col 2 (local 1): B01, B11.
                
                # V_in[row][col]
                # Row for B_j, k_global.
                # B_0,1 (j=0, k_g=1) is B00. Coeff from V_s2[0][r].
                # B_0,2 (j=0, k_g=2) is B01. Coeff from V_s2[1][r].
                # B_1,1 (j=1, k_g=1) is B10. Coeff from V_s2[2][r].
                # B_1,2 (j=1, k_g=2) is B11. Coeff from V_s2[3][r].
                
    # Let's explicitly fill V_in for Strassen part
    for r in range(7):
        V_in[0*3+1][4+r] = V_s2[0][r] # B00
        V_in[0*3+2][4+r] = V_s2[1][r] # B01
        V_in[1*3+1][4+r] = V_s2[2][r] # B10
        V_in[1*3+2][4+r] = V_s2[3][r] # B11
        
        # W_in for Strassen part
        # W_s2 rows: C00, C01, C10, C11.
        # Global C cols 1,2.
        # C_1,0 (k=1, i=0) is C00. Coeff W_s2[0][r].
        # C_1,1 (k=1, i=1) is C01. Coeff W_s2[1][r].
        # C_2,0 (k=2, i=0) is C10. Coeff W_s2[2][r].
        # C_2,1 (k=2, i=1) is C11. Coeff W_s2[3][r].
        
        W_in[1*2+0][4+r] = W_s2[0][r]
        W_in[1*2+1][4+r] = W_s2[1][r]
        W_in[2*2+0][4+r] = W_s2[2][r]
        W_in[2*2+1][4+r] = W_s2[3][r]
        
    # Outer 2x2x2 Strassen (Rank 7)
    # We use U_s2, V_s2, W_s2 again for the outer blocks.
    
    # Total Rank = 7 * 11 = 77
    # U_total = U_s2 (Kron) U_in -> 16x77
    # V_total = V_s2 (Kron) V_in -> 24x77
    # W_total = W_s2 (Kron) W_in -> 24x77
    
    U_total = kron(U_s2, U_in)
    V_total = kron(V_s2, V_in)
    W_total = kron(W_s2, W_in)
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
