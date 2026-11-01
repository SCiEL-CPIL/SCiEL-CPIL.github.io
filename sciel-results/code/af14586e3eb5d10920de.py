import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5

    # We decompose the 2x3 x 3x5 multiplication into:
    # Two blocks of 2 columns each (cols 0,1 and cols 2,3) handled by a rank-11 2x3x2 decomposition
    # One block of 1 column (col 4) handled by a rank-6 2x3x1 decomposition (schoolbook)
    # Total rank: 11 + 11 + 6 = 28

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # (6, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (15, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (10, 28)

    def add_strassen_block(col_offset, b_col_start):
        """
        Add a 2x3x2 Strassen-based decomposition for columns [b_col_start, b_col_start+1] of B.
        Returns the number of columns used (always 11).
        """
        # Strassen for 2x2x2 uses 7 multiplications
        # Plus 4 outer-product terms for the third row/col interaction
        # Total: 11 terms

        col = col_offset

        # Strassen terms for A_L (rows 0,1 of A; cols 0,1) and B_L (rows 0,1 of B; cols b_col_start, b_col_start+1)
        # A_L[i,j] for i in {0,1}, j in {0,1}
        # B_L[j,k] for j in {0,1}, k in {b_col_start, b_col_start+1}

        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)

        # B indices: V row = j*p + k, where j is row of B (0..2), k is col of B (0..4)
        # For B_L: j in {0,1}, k in {b_col_start, b_col_start+1}
        # V index for B_L[j][k_local] where k_local in {0,1}: row = j*p + (b_col_start + k_local)

        # M1: (A00 + A11) * (B00 + B11)
        U[0, col] = 1.0   # A[0,0]
        U[3, col] = 1.0   # A[1,1]  (index: 1*3+1 = 4? Wait. A index: i*m+j. i=1, j=1 -> 3+1=4)
        # Correction: A index is i*m + j.
        # A[0,0] -> 0*3+0 = 0
        # A[0,1] -> 0*3+1 = 1
        # A[1,0] -> 1*3+0 = 3
        # A[1,1] -> 1*3+1 = 4
        # A[0,2] -> 0*3+2 = 2
        # A[1,2] -> 1*3+2 = 5

        # Let's redo indices carefully.
        # U row index = i*m + j
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=0, j=2 -> 2
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # i=1, j=2 -> 5

        # V row index = j*p + k
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=0, k=2 -> 2
        # j=0, k=3 -> 3
        # j=0, k=4 -> 4
        # j=1, k=0 -> 5
        # j=1, k=1 -> 6
        # j=1, k=2 -> 7
        # j=1, k=3 -> 8
        # j=1, k=4 -> 9
        # j=2, k=0 -> 10
        # j=2, k=1 -> 11
        # j=2, k=2 -> 12
        # j=2, k=3 -> 13
        # j=2, k=4 -> 14

        # W row index = k*n + i
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # k=2, i=0 -> 4
        # k=2, i=1 -> 5
        # k=3, i=0 -> 6
        # k=3, i=1 -> 7
        # k=4, i=0 -> 8
        # k=4, i=1 -> 9

        # Let's define a helper to add a term
        def add_term(u_indices, v_indices, w_indices, sign=1.0):
            nonlocal col
            for u_idx in u_indices:
                U[u_idx, col] += sign
            for v_idx in v_indices:
                V[v_idx, col] += sign
            for w_idx in w_indices:
                W[w_idx, col] += sign
            col += 1

        # k0 = b_col_start, k1 = b_col_start + 1
        k0 = b_col_start
        k1 = b_col_start + 1

        # V indices for B_L:
        # B[0][k0] -> 0*p + k0 = k0
        # B[0][k1] -> 0*p + k1 = k1
        # B[1][k0] -> 1*p + k0 = p + k0
        # B[1][k1] -> 1*p + k1 = p + k1

        # W indices for C_L (local cols 0,1 of this block, which are global cols k0, k1):
        # C[0][k0] -> k0*n + 0 = k0
        # C[0][k1] -> k1*n + 0 = k1
        # C[1][k0] -> k0*n + 1 = k0 + 1
        # C[1][k1] -> k1*n + 1 = k1 + 1

        # M1 = (A00 + A11) * (B0k0 + B1k0) ... wait, Strassen is for 2x2.
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # Here a=A[0][0], b=A[0][1], c=A[1][0], d=A[1][1]
        # e=B[0][k0], f=B[0][k1], g=B[1][k0], h=B[1][k1]
        #
        # C[0][k0] = M1 - M4 + M5 - M7
        # C[0][k1] = M3 + M4
        # C[1][k0] = M2 + M5
        # C[1][k1] = M1 - M2 + M3 + M6

        # M1
        add_term([0, 4], [k0, p + k0], [k0, k0+1, k1, k1+1])
        # Wait, W indices:
        # C[0][k0] is W[k0*n+0] = W[k0]
        # C[0][k1] is W[k1*n+0] = W[k1]
        # C[1][k0] is W[k0*n+1] = W[k0+1]
        # C[1][k1] is W[k1*n+1] = W[k1+1]
        
        # Let's re-implement add_term to be more explicit about W contributions per term.
        # Actually, it's easier to just build the 11 terms manually.

        # Reset col to col_offset for this block
        col = col_offset

        # Term 1: M1 = (A00 + A11) * (B0k0 + B1k0) ... No, (B0k0 + B1k0) is not right.
        # M1 = (A00 + A11) * (B0k0 + B1k0) is wrong. M1 = (A00 + A11) * (B0k0 + B1k0) ???
        # Standard Strassen: M1 = (a+d)(e+h). e=B0k0, h=B1k1.
        # So M1 = (A00 + A11) * (B0k0 + B1k1)
        
        # Let's list the 7 Strassen terms and 4 outer product terms.

        # --- Strassen Terms ---
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        # U: A00(0), A11(4)
        # V: B0k0(k0), B1k1(p+k1)
        # W: C0k0(k0), C0k1(k1), C1k0(k0+1), C1k1(k1+1) with coefficients from C formulas.
        # This approach of grouping is error-prone. Let's define each of the 11 multiplications explicitly.

        # Multiplication 1: (A00 + A11) * (B0k0 + B1k1)
        U[0, col] += 1.0; U[4, col] += 1.0
        V[k0, col] += 1.0; V[p+k1, col] += 1.0
        # Contribution to C:
        # C0k0 += 1, C0k1 += 1, C1k0 += 1, C1k1 += 1  (from M1 in C formulas)
        # C0k0 = M1 - M4 + M5 - M7
        # C0k1 = M3 + M4
        # C1k0 = M2 + M5
        # C1k1 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C0k0 and +1 to C1k1.
        W[k0, col] += 1.0
        W[k1+1, col] += 1.0
        col += 1

        # Multiplication 2: (A10 + A11) * B0k0
        U[3, col] += 1.0; U[4, col] += 1.0
        V[k0, col] += 1.0
        # M2 contributes +1 to C1k0
        W[k0+1, col] += 1.0
        col += 1

        # Multiplication 3: A00 * (B0k1 - B1k1)
        U[0, col] += 1.0
        V[k1, col] += 1.0; V[p+k1, col] -= 1.0
        # M3 contributes +1 to C0k1 and +1 to C1k1
        W[k1, col] += 1.0
        W[k1+1, col] += 1.0
        col += 1

        # Multiplication 4: A11 * (B1k0 - B0k0)
        U[4, col] += 1.0
        V[p+k0, col] += 1.0; V[k0, col] -= 1.0
        # M4 contributes -1 to C0k0 and +1 to C0k1
        W[k0, col] -= 1.0
        W[k1, col] += 1.0
        col += 1

        # Multiplication 5: (A00 + A01) * B1k1
        U[0, col] += 1.0; U[1, col] += 1.0
        V[p+k1, col] += 1.0
        # M5 contributes +1 to C0k0 and +1 to C1k0
        W[k0, col] += 1.0
        W[k0+1, col] += 1.0
        col += 1

        # Multiplication 6: (A10 - A00) * (B0k0 + B0k1)
        U[3, col] += 1.0; U[0, col] -= 1.0
        V[k0, col] += 1.0; V[k1, col] += 1.0
        # M6 contributes +1 to C1k1
        W[k1+1, col] += 1.0
        col += 1

        # Multiplication 7: (A01 - A11) * (B1k0 + B1k1)
        U[1, col] += 1.0; U[4, col] -= 1.0
        V[p+k0, col] += 1.0; V[p+k1, col] += 1.0
        # M7 contributes -1 to C0k0
        W[k0, col] -= 1.0
        col += 1

        # --- Outer Product Terms for A[:,2] * B[2,:] ---
        # A02 * B2k0 -> C0k0
        # A02 * B2k1 -> C0k1
        # A12 * B2k0 -> C1k0
        # A12 * B2k1 -> C1k1

        # Multiplication 8: A02 * B2k0
        U[2, col] += 1.0
        V[2*p+k0, col] += 1.0  # j=2, k=k0 -> 2*p + k0
        W[k0, col] += 1.0
        col += 1

        # Multiplication 9: A02 * B2k1
        U[2, col] += 1.0
        V[2*p+k1, col] += 1.0
        W[k1, col] += 1.0
        col += 1

        # Multiplication 10: A12 * B2k0
        U[5, col] += 1.0
        V[2*p+k0, col] += 1.0
        W[k0+1, col] += 1.0
        col += 1

        # Multiplication 11: A12 * B2k1
        U[5, col] += 1.0
        V[2*p+k1, col] += 1.0
        W[k1+1, col] += 1.0
        col += 1

        return col - col_offset  # Should be 11

    # Block 1: columns 0, 1
    add_strassen_block(0, 0)
    
    # Block 2: columns 2, 3
    add_strassen_block(11, 2)

    # Block 3: column 4 (schoolbook 2x3x1)
    # 6 terms: A[i][j] * B[j][4] for i in {0,1}, j in {0,1,2}
    col = 22
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] += 1.0
            V[j*p + 4, col] += 1.0
            W[4*n + i, col] += 1.0  # k=4, i
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
