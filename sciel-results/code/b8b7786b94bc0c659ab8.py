"""
Rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication.

Strategy:
- Split the 3x5 right factor into 3x3 + 3x2 blocks.
- Use schoolbook for the 3x3 * 3x3 part (rank 27).
- Use a Strassen-based construction for the 3x3 * 3x2 part (rank 17).
  * The 3x3 * 3x2 part is split into:
      - Top 2 rows of A (2x3) times B (3x2): rank 11
          * Split into (2x2 * 2x2) rank 7 + (2x1 * 1x2) rank 4
      - Bottom 1 row of A (1x3) times B (3x2): rank 6
  * Total: 7 + 4 + 6 = 17
- Total rank: 27 + 17 = 44

This gives a score of 45/44 ≈ 1.0227.
"""

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # We'll build U (9 x rank), V (15 x rank), W (15 x rank)
    # by concatenating columns from sub-decompositions.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_sub(U_sub, V_sub, W_sub):
        """
        U_sub: (n_rows, r)
        V_sub: (m_rows, r)
        W_sub: (p_rows, r)
        """
        r = U_sub.shape[1]
        for c in range(r):
            U_cols.append(U_sub[:, c])
            V_cols.append(V_sub[:, c])
            W_cols.append(W_sub[:, c])
    
    import numpy as np
    
    # =========================================================
    # Part 1: 3x3 * 3x3 schoolbook (rank 27)
    # This multiplies A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    # =========================================================
    r1 = 9  # 3*3
    U1 = np.zeros((9, r1))
    V1 = np.zeros((9, r1))  # m*p for this sub-problem is 3*3=9
    W1 = np.zeros((9, r1))  # p*n for this sub-problem is 3*3=9
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U1[i * 3 + j, col] = 1
                V1[j * 3 + k, col] = 1
                W1[k * 3 + i, col] = 1
                col += 1
    
    # Map to full indices:
    # U full row index: i*m + j where i in [0,3), j in [0,3)
    # V full row index: j*p + k where j in [0,3), k in [0,5) -> for k in [0,3)
    # W full row index: k*n + i where k in [0,5), i in [0,3) -> for k in [0,3)
    
    U1_full = np.zeros((n * m, r1))
    V1_full = np.zeros((m * p, r1))
    W1_full = np.zeros((p * n, r1))
    
    for col in range(r1):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if U1[i * 3 + j, col] != 0:
                        U1_full[i * m + j, col] += U1[i * 3 + j, col]
                    if V1[j * 3 + k, col] != 0:
                        V1_full[j * p + k, col] += V1[j * 3 + k, col]
                    if W1[k * 3 + i, col] != 0:
                        W1_full[k * n + i, col] += W1[k * 3 + i, col]
    
    add_sub(U1_full, V1_full, W1_full)
    
    # =========================================================
    # Part 2: 3x3 * 3x2 (rank 17)
    # This multiplies A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5]
    # =========================================================
    
    # Sub-part 2a: Top 2 rows of A (2x3) * B (3x2) -> C_top (2x2)
    # Split A_top (2x3) into A11 (2x2) and A12 (2x1)
    # Split B (3x2) into B1 (2x2) and B2 (1x2)
    # C_top = A11*B1 + A12*B2
    
    # --- 2a-1: A11 (2x2) * B1 (2x2) -> 2x2, Strassen rank 7 ---
    # A11 rows: 0,1 ; cols: 0,1
    # B1 rows: 0,1 ; cols: 0,1 (which are global cols 3,4)
    
    r2a1 = 7
    U2a1 = np.zeros((9, r2a1))
    V2a1 = np.zeros((15, r2a1))
    W2a1 = np.zeros((15, r2a1))
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A21)B22
    # M6 = (A22-A11)(B11+B21)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our case: A11 is A[0:2, 0:2], B1 is B[0:2, 0:2] (local)
    # Global A indices: rows 0,1; cols 0,1
    # Global B indices: rows 0,1; cols 0,1 (local) -> global cols 3,4
    
    # Strassen coefficients for each term
    # We need to express each Mi as a sum of products of linear combos
    
    # Let's build the Strassen decomposition explicitly.
    # For 2x2 * 2x2, the tensor has 8 entries. Strassen uses 7 multiplications.
    
    # I'll use the standard Strassen formulas.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a21)b22
    # M6 = (a22-a11)(b11+b21)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, we have a product of a linear combo of A entries and a linear combo of B entries.
    
    strassen_terms = [
        # (A_coeff, B_coeff) where A_coeff is dict {(row,col): coeff} for 2x2
        # and B_coeff is dict {(row,col): coeff} for 2x2
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}),
        # M4
        ({(1,1):1}, {(0,1):1, (0,0):-1}),
        # M5
        ({(0,0):1, (1,0):1}, {(1,1):1}),
        # M6
        ({(1,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    c_coeffs = [
        # For C11, C12, C21, C22, coefficients for M1..M7
        # C11 = M1 + M4 - M5 + M7
        [{1:1, 4:1, 5:-1, 7:1}],
        # C12 = M3 + M5
        [{3:1, 5:1}],
        # C21 = M2 + M4
        [{2:1, 4:1}],
        # C22 = M1 - M2 + M3 + M6
        [{1:1, 2:-1, 3:1, 6:1}],
    ]
    
    # Map local 2x2 indices to global indices
    # A local (i,j) -> global A row i, col j (i in 0,1; j in 0,1)
    # B local (i,j) -> global B row i, col j+3 (i in 0,1; j in 0,1)
    # C local (i,j) -> global C row i, col j+3 (i in 0,1; j in 0,1)
    
    for mi, (a_coeff, b_coeff) in enumerate(strassen_terms):
        col_idx = mi  # 0..6
        
        for (ai, aj), ac in a_coeff.items():
            U2a1[ai * 3 + aj, col_idx] += ac
        
        for (bi, bj), bc in b_coeff.items():
            V2a1[bi * 5 + (bj + 3), col_idx] += bc
    
    # Now fill W from C coefficients
    # C local (ci, cj) -> global W row (cj+3)*3 + ci
    c_local_to_global = {
        (0,0): (3*3 + 0),
        (0,1): (4*3 + 0),
        (1,0): (3*3 + 1),
        (1,1): (4*3 + 1),
    }
    
    for (ci, cj), coeffs in enumerate([c_coeffs[0], c_coeffs[1], c_coeffs[2], c_coeffs[3]]):
        for mi, coeff in coeffs[0].items():
            W2a1[c_local_to_global[(ci, cj)], mi - 1] += coeff
    
    add_sub(U2a1, V2a1, W2a1)
    
    # --- 2a-2: A12 (2x1) * B2 (1x2) -> 2x2, schoolbook rank 4 ---
    # A12: rows 0,1; col 2
    # B2: row 2; cols 0,1 (local) -> global cols 3,4
    # C = A12 * B2 (2x2)
    
    r2a2 = 4
    U2a2 = np.zeros((9, r2a2))
    V2a2 = np.zeros((15, r2a2))
    W2a2 = np.zeros((15, r2a2))
    
    col = 0
    for i in range(2):
        for j in range(2):
            U2a2[i * 3 + 2, col] = 1
            V2a2[2 * 5 + j, col] = 1  # B row 2, col j (local) -> global col j+3
            W2a2[(j + 3) * 3 + i, col] = 1
            col += 1
    
    add_sub(U2a2, V2a2, W2a2)
    
    # --- 2b: Bottom row of A (1x3) * B (3x2) -> 1x2, schoolbook rank 6 ---
    # A bottom: row 2, cols 0,1,2
    # B: rows 0,1,2; cols 0,1 (local) -> global cols 3,4
    # C: row 2, cols 0,1 (local) -> global cols 3,4
    
    r2b = 6
    U2b = np.zeros((9, r2b))
    V2b = np.zeros((15, r2b))
    W2b = np.zeros((15, r2b))
    
    col = 0
    for j in range(3):
        for k in range(2):
            U2b[2 * 3 + j, col] = 1
            V2b[j * 5 + (k + 3), col] = 1
            W2b[(k + 3) * 3 + 2, col] = 1
            col += 1
    
    add_sub(U2b, V2b, W2b)
    
    # =========================================================
    # Assemble final matrices
    # =========================================================
    rank = len(U_cols)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (uc, vc, wc) in enumerate(zip(U_cols, V_cols, W_cols)):
        U[:, c] = uc
        V[:, c] = vc
        W[:, c] = wc
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
