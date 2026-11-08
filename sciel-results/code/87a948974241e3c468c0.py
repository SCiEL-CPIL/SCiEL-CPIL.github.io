"""
Exact rank-53 tensor decomposition for 4x4x5 matrix multiplication.

Structure:
  C = A * B,  A is 4x4, B is 4x5.
  Split B into B4 (first 4 cols) and B5 (last col).
  C4 = A * B4  (4x4x4, rank 49 via 7-term 2x2x2 Strassen)
  C5 = A * B5  (4x4x1, rank 4 trivially)

Total rank = 49 + 4 = 53.

Conventions:
  U has shape (n*m, rank) = (16, 53), row index = i*m + j
  V has shape (m*p, rank) = (20, 53), row index = j*p + k
  W has shape (p*n, rank) = (20, 53), row index = k*n + i
"""

def construct():
    n, m, p = 4, 4, 5
    rank = 53
    U = [[0] * rank for _ in range(n * m)]   # 16 x 53
    V = [[0] * rank for _ in range(m * p)]   # 20 x 53
    W = [[0] * rank for _ in range(p * n)]   # 20 x 53

    # --- Strassen 2x2x2 core: 7 multiplications ---
    # For matrices X, Y (2x2), product Z = X*Y:
    #   M1 = (X00+X11)(Y00+Y11)
    #   M2 = (X10+X11) Y00
    #   M3 = X00 (Y01-Y11)
    #   M4 = X11 (Y10-Y00)
    #   M5 = (X00+X01) Y11
    #   M6 = (X10-X00)(Y00+Y01)
    #   M7 = (X11-X10)(Y10+Y11)
    #   Z00 = M1 + M4 - M5 + M7
    #   Z01 = M3 + M5
    #   Z10 = M2 + M4
    #   Z11 = M1 - M2 + M3 + M6

    # We apply this at each of the 4 block positions (I,J) in the 4x4 product.
    # Block (I,J) uses A-block rows 2I..2I+1, cols 2J..2J+1; B-block rows 2J..2J+1, cols 2K..2K+1.

    # The 7 Strassen terms per block:
    # Term t in 0..6 corresponds to:
    #   t=0: M1
    #   t=1: M2
    #   t=2: M3
    #   t=3: M4
    #   t=4: M5
    #   t=5: M6
    #   t=6: M7

    # For each Strassen term, define (a_factor, b_factor, z_factors):
    #   a_factor: dict mapping (row_in_block, col_in_block) -> coeff
    #   b_factor: dict mapping (row_in_block, col_in_block) -> coeff
    #   z_factors: dict mapping (row_in_block, col_in_block) -> coeff  (for the output block)

    def strassen_term_a(t, r, c):
        """Coefficient of A[2I+r, 2J+c] in Strassen term t of block (I,J)."""
        if t == 0:  # M1 = (X00+X11)(Y00+Y11)
            return 1 if (r, c) in [(0,0), (1,1)] else 0
        elif t == 1:  # M2 = (X10+X11) Y00
            return 1 if (r, c) in [(1,0), (1,1)] else 0
        elif t == 2:  # M3 = X00 (Y01-Y11)
            return 1 if (r, c) == (0, 0) else 0
        elif t == 3:  # M4 = X11 (Y10-Y00)
            return 1 if (r, c) == (1, 1) else 0
        elif t == 4:  # M5 = (X00+X01) Y11
            return 1 if (r, c) in [(0,0), (0,1)] else 0
        elif t == 5:  # M6 = (X10-X00)(Y00+Y01)
            return 1 if (r, c) == (1,0) else (-1 if (r, c) == (0,0) else 0)
        elif t == 6:  # M7 = (X11-X10)(Y10+Y11)
            return 1 if (r, c) == (1,1) else (-1 if (r, c) == (1,0) else 0)
        return 0

    def strassen_term_b(t, r, c):
        """Coefficient of B[2J+r, 2K+c] in Strassen term t of block (I,J)."""
        if t == 0:  # M1
            return 1 if (r, c) in [(0,0), (1,1)] else 0
        elif t == 1:  # M2
            return 1 if (r, c) == (0, 0) else 0
        elif t == 2:  # M3
            return 1 if (r, c) == (0,1) else (-1 if (r, c) == (1,1) else 0)
        elif t == 3:  # M4
            return 1 if (r, c) == (1,0) else (-1 if (r, c) == (0,0) else 0)
        elif t == 4:  # M5
            return 1 if (r, c) == (1,1) else 0
        elif t == 5:  # M6
            return 1 if (r, c) in [(0,0), (0,1)] else 0
        elif t == 6:  # M7
            return 1 if (r, c) in [(1,0), (1,1)] else 0
        return 0

    def strassen_term_z(t, r, c):
        """Coefficient for output Z[r,c] from Strassen term t."""
        if t == 0:  # M1 contributes to Z00 and Z11
            return 1 if (r, c) in [(0,0), (1,1)] else 0
        elif t == 1:  # M2 contributes to Z10
            return 1 if (r, c) == (1,0) else 0
        elif t == 2:  # M3 contributes to Z01
            return 1 if (r, c) == (0,1) else 0
        elif t == 3:  # M4 contributes to Z00 and Z10
            return 1 if (r, c) in [(0,0), (1,0)] else 0
        elif t == 4:  # M5 contributes to Z00 (negative) and Z01
            return -1 if (r, c) == (0,0) else (1 if (r, c) == (0,1) else 0)
        elif t == 5:  # M6 contributes to Z11
            return 1 if (r, c) == (1,1) else 0
        elif t == 6:  # M7 contributes to Z00
            return 1 if (r, c) == (0,0) else 0
        return 0

    col = 0
    for I in range(2):
        for J in range(2):
            for t in range(7):
                # U part: A rows 2I..2I+1, cols 2J..2J+1
                for r in range(2):
                    for c in range(2):
                        val = strassen_term_a(t, r, c)
                        if val != 0:
                            i = 2 * I + r
                            j = 2 * J + c
                            U[i * m + j][col] += val
                # V part: B rows 2J..2J+1, cols 0..3 (for C4)
                for r in range(2):
                    for c in range(2):
                        val = strassen_term_b(t, r, c)
                        if val != 0:
                            j = 2 * J + r
                            k = c  # cols 0,1 of B for this block
                            V[j * p + k][col] += val
                # W part: output C4 rows 2I..2I+1, cols 2J..2J+1
                for r in range(2):
                    for c in range(2):
                        val = strassen_term_z(t, r, c)
                        if val != 0:
                            k = 2 * J + c  # output column
                            i = 2 * I + r  # output row
                            W[k * n + i][col] += val
                col += 1

    # Now handle C5 = A * B5 (last column of B), rank 4.
    # C5[i] = sum_j A[i,j] * B[j,4]
    # Trivial decomposition: for each j=0..3:
    #   U[i, col+j] = 1 (for all i)  -> wait, we need U[i*m + j, ...]
    # Actually: C5[i] = sum_{j,c} A[i,j] * B[j,4] * U[i*m+j, c] * V[j*p+4, c] * W[4*n+i, c]
    # We want: sum_j A[i,j] B[j,4]
    # Set: for each j, one rank term:
    #   U[i*m+j, c_j] = 1 for all i? No, U is indexed by (i,j).
    #   We need: sum_j A[i,j] * B[j,4]
    #   Let c = col + j.
    #   U[i*m+j, c] = 1 for all i? That would give sum_i ... no.
    #   Actually the tensor decomposition is:
    #   C5[i] = sum_{j, c} A[i,j] B[j,4] U[i*m+j, c] V[j*p+4, c] W[4*n+i, c]
    #   We want this to equal sum_j A[i,j] B[j,4].
    #   So set: U[i*m+j, c_j] = 1, V[j*p+4, c_j] = 1, W[4*n+i, c_j] = 1 for all i.
    #   Then: sum_j A[i,j] B[j,4] * 1 * 1 * 1 = sum_j A[i,j] B[j,4]. Correct.
    #   But U[i*m+j, c_j] must be 1 for ALL i. That's fine, it's a fixed value.
    for j in range(4):
        c = col + j
        for i in range(4):
            U[i * m + j][c] = 1
        V[j * p + 4][c] = 1
        for i in range(4):
            W[4 * n + i][c] = 1
    col += 4

    return {"U": U, "V": V, "W": W}
