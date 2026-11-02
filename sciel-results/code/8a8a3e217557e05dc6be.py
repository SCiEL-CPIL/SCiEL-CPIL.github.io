"""Matrix multiplication 2x2 x 2x12 via block-wise Strassen (rank 42)."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x2-by-2x12 multiplication tensor.
    Split the 2x12 matrix into six 2x2 blocks along the p=12 dimension (6 blocks of size 2).
    Apply the 7-term Strassen identity to each 2x2 x 2x2 block multiplication.
    Total rank = 6 blocks * 7 = 42.

    U has shape (n*m, rank) = (4, 42)
    V has shape (m*p, rank) = (24, 42)
    W has shape (p*n, rank) = (24, 42)  with trace convention W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 6 blocks of width 2
    assert p % 2 == 0, "p must be even for this decomposition"
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42

    # Strassen 7 terms for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # A = [[A11, A12], [A21, A22]]  (2x2)
    # B = [[B11, B12], [B21, B22]]  (2x2)
    #
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    #
    # For our tensor:
    # A is the 2x2 left matrix (n x m = 2 x 2)
    # B is a 2x2 block of the right matrix (m x 2, since each block covers 2 columns of the 2x12)
    # C is a 2x2 block of the output (2 x 2, covering 2 rows of the 2x12 output... wait)
    #
    # Actually: (2x2) * (2x12) = (2x12)
    # Split right matrix into 6 blocks of (2x2).
    # Output is 2x12, split into 6 blocks of (2x2).
    #
    # Left matrix A = [[a00, a01], [a10, a11]] indexed by row i (0,1) and col j (0,1)
    # In U: row index = i*m + j = i*2 + j
    #
    # Right block b (covers columns 2b, 2b+1 of the 2x12):
    # B = [[b00, b01], [b10, b11]] where b00 = element (0, 2b), b01 = (0, 2b+1), b10 = (1, 2b), b11 = (1, 2b+1)
    # In V: row index = j*p + k = j*12 + k
    # So B11 = V[0*12 + 2b], B12 = V[0*12 + 2b+1], B21 = V[1*12 + 2b], B22 = V[1*12 + 2b+1]
    #
    # Output block b (covers columns 2b, 2b+1 of the 2x12 output):
    # C = [[c00, c01], [c10, c11]] where c00 = (0, 2b), c01 = (0, 2b+1), c10 = (1, 2b), c11 = (1, 2b+1)
    # In W: W[k*n + i, r] where k is the column index in output, i is the row index
    # So C11 (row 0, col 0 of block) = W[(2b)*2 + 0] = W[4b]
    #    C12 (row 0, col 1 of block) = W[(2b+1)*2 + 0] = W[4b+1]
    #    C21 (row 1, col 0 of block) = W[(2b)*2 + 1] = W[4b+2]
    #    C22 (row 1, col 1 of block) = W[(2b+1)*2 + 1] = W[4b+3]

    # Strassen terms as (U_rows_with_signs, V_rows_with_signs)
    # U row indices: A11=0, A12=1, A21=2, A22=3  (i*m+j)
    # For a given block b, V row indices:
    #   B11 = 0*12 + 2b = 2b
    #   B12 = 0*12 + 2b+1 = 2b+1
    #   B21 = 1*12 + 2b = 12+2b
    #   B22 = 1*12 + 2b+1 = 12+2b+1

    # Define the 7 Strassen terms: each is a list of (u_idx, sign_u) and (v_idx, sign_v)
    # We'll store for each term: list of (u_row, sign) and (v_row, sign)
    # Then W contributions are determined by which C element each term feeds into.

    # Term definitions (u_combination, v_combination):
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A11-A22) * (B21+B22)  -- note: A12-A22 in standard, but let me recheck

    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each block b and each term t, we add contributions to U, V, and W.
    # U and V get the factor values; W gets the combination signs.

    for b in range(num_blocks):
        base = b * 7
        B11_v = 2 * b
        B12_v = 2 * b + 1
        B21_v = 12 + 2 * b
        B22_v = 12 + 2 * b + 1

        # For each term, define:
        #   u_parts: list of (u_row, sign)
        #   v_parts: list of (v_row, sign)
        #   w_parts: list of (w_row, sign)  -- which C elements this term contributes to

        terms = [
            # (u_parts, v_parts, w_parts)
            # M1 = (A11+A22)(B11+B22)
            ([(0, 1), (3, 1)], [(B11_v, 1), (B22_v, 1)],
             [(4*b, 1), (4*b+2, -1), (4*b+3, 1)]),  # C11+, C21-, C22+
            # M2 = (A21+A22)B11
            ([(2, 1), (3, 1)], [(B11_v, 1)],
             [(4*b+2, 1), (4*b+3, -1)]),  # C21+, C22-
            # M3 = A11(B12-B22)
            ([(0, 1)], [(B12_v, 1), (B22_v, -1)],
             [(4*b+1, 1), (4*b+3, 1)]),  # C12+, C22+
            # M4 = A22(B21-B11)
            ([(3, 1)], [(B21_v, 1), (B11_v, -1)],
             [(4*b, 1), (4*b+2, 1)]),  # C11+, C21+
            # M5 = (A11+A12)B22
            ([(0, 1), (1, 1)], [(B22_v, 1)],
             [(4*b, -1), (4*b+1, 1)]),  # C11-, C12+
            # M6 = (A21-A11)(B11+B12)
            ([(2, 1), (0, -1)], [(B11_v, 1), (B12_v, 1)],
             [(4*b+3, 1)]),  # C22+
            # M7 = (A12-A22)(B21+B22)
            ([(1, 1), (3, -1)], [(B21_v, 1), (B22_v, 1)],
             [(4*b, 1)]),  # C11+
        ]

        for t, (u_parts, v_parts, w_parts) in enumerate(terms):
            col = base + t
            for u_row, sign in u_parts:
                U[u_row][col] += sign
            for v_row, sign in v_parts:
                V[v_row][col] += sign
            for w_row, sign in w_parts:
                W[w_row][col] += sign

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
