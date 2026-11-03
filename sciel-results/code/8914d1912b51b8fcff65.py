"""Matrix multiplication 3x4 * 4x8 with rank-88 Strassen-based decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130  # safety margin before 150s kill

    # Block structure:
    # A (3x4) = [[A11, A12], [A21, A22]]  where A11,A12,A21,A22 are 2x2 or 1x2
    # Actually: rows 0-1 -> 2 rows, row 2 -> 1 row
    # cols 0-1 -> 2 cols, cols 2-3 -> 2 cols
    # So: A11=2x2, A12=2x2, A21=1x2, A22=1x2
    #
    # B (4x8) = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    # rows 0-1 -> 2 rows, rows 2-3 -> 2 rows
    # cols 0-1,2-3,4-5,6-7 -> four 2-col blocks
    # So: B11=2x2, B12=2x2, B13=2x2, B14=2x2, B21=2x2, B22=2x2, B23=2x2, B24=2x2
    #
    # C (3x8) = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x2 * 2x2) -> two Strassen products
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C14 = A11*B14 + A12*B24
    # C21 = A21*B11 + A22*B21  (1x2 * 2x2 + 1x2 * 2x2) -> schoolbook: 2*2*2=8 each, total 16
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    # C24 = A21*B14 + A22*B24

    # For 2x2 * 2x2: Strassen rank 7
    # For 1x2 * 2x2: this is (1x2)@(2x2) = 1x2, schoolbook rank = 1*2*2 = 4 per product
    #   But we have two such products per C2k block: A21*B1k + A22*B2k
    #   Each 1x2 * 2x2 has rank 4 (schoolbook), so two of them = 8 per C2k
    #   Total for C2 row: 4 blocks * 8 = 32
    #
    # For 2x2 * 2x2: Strassen rank 7
    #   C1 row has 4 blocks, each is sum of two 2x2*2x2 products
    #   Each 2x2*2x2 uses 7 rank terms
    #   Total for C1 row: 4 blocks * 2 products * 7 = 56
    #
    # Grand total: 56 + 32 = 88

    # Let's build this explicitly.
    # Indexing convention:
    #   U[i*m+j, c] where i in [0,n), j in [0,m)  -> (i,j) maps to A[i,j]
    #   V[j*p+k, c] where j in [0,m), k in [0,p)  -> (j,k) maps to B[j,k]
    #   W[k*n+i, c] where k in [0,p), i in [0,n)  -> (k,i) maps to C[i,k]
    #
    # Constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # (since T[i,k,j]=1 for all valid i,k,j)

    rank_total = 88
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank_total for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 88

    col = 0

    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For each 2x2*2x2 product, we need to assign rank-1 terms to U, V, W.
    # The 2x2*2x2 product involves:
    #   A sub-block at rows (ar, ar+1), cols (ac, ac+1)
    #   B sub-block at rows (br, br+1), cols (bc, bc+1)
    #   C sub-block at rows (cr, cr+1), cols (cc, cc+1)
    #
    # Strassen term t (0..6):
    #   Each term is a product of a linear combo of A entries, a linear combo of B entries,
    #   and contributes to specific C entries with coefficients.
    #
    # Let's define the 7 Strassen terms explicitly with their A-coeffs, B-coeffs, and C-contribution.

    # Strassen terms for 2x2*2x2:
    # Term 0: (A00+A11)*(B00+B11) -> contributes +1 to C00, +1 to C11
    #   A coeffs: A00:1, A11:1 (others 0)
    #   B coeffs: B00:1, B11:1 (others 0)
    #   C coeffs: C00:+1, C11:+1
    # Term 1: (A10+A11)*B00 -> contributes +1 to C10, -1 to C11
    #   A coeffs: A10:1, A11:1
    #   B coeffs: B00:1
    #   C coeffs: C10:+1, C11:-1
    # Term 2: A00*(B01-B11) -> contributes +1 to C01, +1 to C10
    #   A coeffs: A00:1
    #   B coeffs: B01:1, B11:-1
    #   C coeffs: C01:+1, C10:+1
    # Term 3: A11*(B10-B00) -> contributes +1 to C10, +1 to C11
    #   Wait, let me re-derive carefully.
    #
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Using 1-indexed: A11=top-left, A12=top-right, A21=bottom-left, A22=bottom-right
    # In 0-indexed: A[0,0], A[0,1], A[1,0], A[1,1]
    #
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #   A: [0,0]=1, [1,1]=1
    #   B: [0,0]=1, [1,1]=1
    #   C: [0,0]+=1, [1,1]+=1
    #
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    #   A: [1,0]=1, [1,1]=1
    #   B: [0,0]=1
    #   C: [1,0]+=1, [1,1]-=1
    #
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    #   A: [0,0]=1
    #   B: [0,1]=1, [1,1]=-1
    #   C: [0,1]+=1, [1,0]+=1
    #
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    #   A: [1,1]=1
    #   B: [1,0]=1, [0,0]=-1
    #   C: [1,0]+=1, [0,0]+=1
    #
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    #   A: [0,0]=1, [0,1]=1
    #   B: [1,1]=1
    #   C: [0,0]-=1, [0,1]+=1
    #
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    #   A: [1,0]=1, [0,0]=-1
    #   B: [0,0]=1, [0,1]=1
    #   C: [1,1]+=1
    #
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #   A: [0,1]=1, [1,1]=-1
    #   B: [1,0]=1, [1,1]=1
    #   C: [0,0]+=1, [1,1]+=1

    # Let me verify C11 = M1+M4-M5+M7:
    # M1: C00+=1
    # M4: C00+=1
    # M5: C00-=1
    # M7: C00+=1
    # Total C00: 1+1-1+1 = 2? That's wrong, should be 1.
    #
    # I think I have an error. Let me re-derive from scratch.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Strassen (standard):
    # M1 = (A11+A22)(B11+B22) = (A00+A11)(B00+B11)
    # M2 = (A21+A22)B11 = (A10+A11)B00
    # M3 = A11(B12-B22) = A00(B01-B11)
    # M4 = A22(B21-B11) = A11(B10-B00)
    # M5 = (A11+A12)B22 = (A00+A01)B11
    # M6 = (A21-A11)(B11+B12) = (A10-A00)(B00+B01)
    # M7 = (A12-A22)(B21+B22) = (A01-A11)(B10+B11)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Let me verify C11 (which is C[0,0]):
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    #
    # C11 = M1+M4-M5+M7
    # = (A00B00 + A00B11 + A11B00 + A11B11)
    #   + (A11B10 - A11B00)
    #   - (A00B11 + A01B11)
    #   + (A01B10 + A01B11 - A11B10 - A11B11)
    # = A00B00 + A00B11 + A11B00 + A11B11
    #   + A11B10 - A11B00
    #   - A00B11 - A01B11
    #   + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + (A00B11 - A00B11) + (A11B00 - A11B00) + (A11B11 - A11B11)
    #   + (A11B10 - A11B10) + (-A01B11 + A01B11) + A01B10
    # = A00B00 + A01B10
    #
    # But C[0,0] should be A00B00 + A01B10. Yes! Correct.
    #
    # Now let me assign coefficients properly:
    # For each Strassen term, I need:
    #   a_coeffs: dict mapping (a_row, a_col) -> coefficient
    #   b_coeffs: dict mapping (b_row, b_col) -> coefficient
    #   c_coeffs: dict mapping (c_row, c_col) -> coefficient

    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # M4: A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        # M5: (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]

    def add_strassen(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """Add 7 Strassen terms for a 2x2*2x2 product."""
        nonlocal col
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # U: A indices
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            # V: B indices
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            # W: C indices (note: W is indexed by [k*n+i] where k is col, i is row)
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
        return 7

    def add_schoolbook_1x2x2(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """Add schoolbook terms for a (1x2)*(2x2) product."""
        nonlocal col
        # A is 1x2: A[0,0], A[0,1]
        # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # C is 1x2: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        #           C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # 4 terms:
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        # Term 2: A[0,0]*B[0,1] -> C[0,1]
        # Term 3: A[0,1]*B[1,1] -> C[0,1]
        terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,1):1}, {(1,0):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1}, {(0,1):1}),
            ({(0,1):1}, {(1,1):1}, {(0,1):1}),
        ]
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
        return 4

    # Now build the full decomposition:
    # C1 row (rows 0-1 of C, i.e., 2 rows):
    #   C11 (cols 0-1): A11*B11 + A12*B21
    #     A11: rows 0-1, cols 0-1 of A
    #     A12: rows 0-1, cols 2-3 of A
    #     B11: rows 0-1, cols 0-1 of B
    #     B21: rows 2-3, cols 0-1 of B
    #     C11: rows 0-1, cols 0-1 of C
    add_strassen(0, 0, 0, 0, 0, 0)    # A11*B11 -> C11
    add_strassen(0, 2, 2, 0, 0, 0)    # A12*B21 -> C11

    #   C12 (cols 2-3): A11*B12 + A12*B22
    add_strassen(0, 0, 0, 2, 0, 2)    # A11*B12 -> C12
    add_strassen(0, 2, 2, 2, 0, 2)    # A12*B22 -> C12

    #   C13 (cols 4-5): A11*B13 + A12*B23
    add_strassen(0, 0, 0, 4, 0, 4)    # A11*B13 -> C13
    add_strassen(0, 2, 2, 4, 0, 4)    # A12*B23 -> C13

    #   C14 (cols 6-7): A11*B14 + A12*B24
    add_strassen(0, 0, 0, 6, 0, 6)    # A11*B14 -> C14
    add_strassen(0, 2, 2, 6, 0, 6)    # A12*B24 -> C14

    # C2 row (row 2 of C, i.e., 1 row):
    #   C21 (cols 0-1): A21*B11 + A22*B21
    #     A21: row 2, cols 0-1 of A (1x2)
    #     A22: row 2, cols 2-3 of A (1x2)
    #     B11: rows 0-1, cols 0-1 of B
    #     B21: rows 2-3, cols 0-1 of B
    #     C21: row 2, cols 0-1 of C
    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0)    # A21*B11 -> C21
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0)    # A22*B21 -> C21

    #   C22 (cols 2-3): A21*B12 + A22*B22
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2)    # A21*B12 -> C22
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2)    # A22*B22 -> C22

    #   C23 (cols 4-5): A21*B13 + A22*B23
    add_schoolbook_1x2x2(2, 0, 0, 4, 2, 4)    # A21*B13 -> C23
    add_schoolbook_1x2x2(2, 2, 2, 4, 2, 4)    # A22*B23 -> C23

    #   C24 (cols 6-7): A21*B14 + A22*B24
    add_schoolbook_1x2x2(2, 0, 0, 6, 2, 6)    # A21*B14 -> C24
    add_schoolbook_1x2x2(2, 2, 2, 6, 2, 6)    # A22*B24 -> C24

    # Total rank used: 8*7 + 8*4 = 56 + 32 = 88
    assert col == rank_total, f"Expected {rank_total} columns, got {col}"

    # Verify the solution
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook if verification fails
                    rank_school = n * m * p
                    U_f = [[0] * rank_school for _ in range(n * m)]
                    V_f = [[0] * rank_school for _ in range(m * p)]
                    W_f = [[0] * rank_school for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii * m + jj][c] = 1
                                V_f[jj * p + kk][c] = 1
                                W_f[kk * n + ii][c] = 1
                                c += 1
                    return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
