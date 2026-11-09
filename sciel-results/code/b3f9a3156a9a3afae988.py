"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x8 matrix.

    The matrix multiplication tensor T has entries T[i, j, k] for i in [0, n),
    j in [0, m), k in [0, p) where (AB)[i, k] = sum_j A[i, j] B[j, k].
    The decomposition is sum_c U[:, c] * V[:, c] * W[:, c] (outer product).
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    W index: W[k*n + i, r] (trace convention).

    We use Strassen's 2x2 matrix multiplication (rank 7) on the 2x2 blocks
    of A (each 1x2) and B (each 2x4), producing 2x2 result blocks (each 1x4).
    """
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2 multiplication:
    # For X (2x2) and Y (2x2):
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y00-Y10)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # For our problem: A is 2x4, B is 4x8, C = AB is 2x8.
    # We partition A into 2x2 blocks: A[a, a+1] where a in {0,1} for row-block, 
    # and a in {0,1} for col-block (since m=4, we have 2 col-blocks of size 2).
    # B is partitioned into 2x2 blocks: B[2a:2a+2, 2b:2b+2] for a in {0,1}, b in {0,3}.
    # C is partitioned into 2x2 blocks: C[a, 2b:2b+2] for a in {0,1}, b in {0,3}.

    # Actually, let's think more carefully.
    # A is 2 rows x 4 cols. Partition into 2x2 blocks: A[0:2, 0:2] and A[0:2, 2:4].
    # B is 4 rows x 8 cols. Partition into 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8],
    #                                                 B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8].
    # C = AB is 2 rows x 8 cols. C[a, k] = sum_j A[a, j] B[j, k].
    # For each pair of A-col-block (j_block in {0,1}) and B-row-block (j_block in {0,1}),
    # we have a 2x2 * 2x2 = 2x2 multiplication for each B-col-block (k_block in {0,1,2,3}).
    # Total: 2 (A col blocks) * 4 (B col blocks) = 8 independent 2x2*2x2 multiplications.
    # Each uses 7 Strassen terms. Total rank = 8 * 7 = 56.

    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen coefficients for the 7 M terms.
    # Each M term: (coeff for A00, coeff for A01, coeff for A10, coeff for A11,
    #               coeff for B00, coeff for B01, coeff for B10, coeff for B11)
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11)
        (1, 0, 0, 1, 1, 0, 0, 1),
        # M2: (A10+A11)*B00
        (0, 0, 1, 1, 1, 0, 0, 0),
        # M3: A00*(B10-B11)
        (1, 0, 0, 0, 0, 0, 1, -1),
        # M4: A11*(B00-B10)
        (0, 0, 0, 1, 1, 0, -1, 0),
        # M5: (A00+A01)*B11
        (1, 1, 0, 0, 0, 0, 0, 1),
        # M6: (A10-A00)*(B00+B01)
        (-1, 0, 1, 0, 1, 1, 0, 0),
        # M7: (A01-A11)*(B10+B11)
        (0, 1, 0, -1, 0, 0, 1, 1),
    ]

    # Strassen combination coefficients:
    # Z00 = M1 + M4 - M5 + M7  => (1, 0, 0, 1, -1, 0, 1)
    # Z01 = M3 + M5            => (0, 0, 1, 0, 1, 0, 0)
    # Z10 = M2 + M4            => (0, 1, 0, 1, 0, 0, 0)
    # Z11 = M1 - M2 + M3 + M6  => (1, -1, 1, 0, 0, 1, 0)
    z_coeffs = [
        (1, 0, 0, 1, -1, 0, 1),   # Z00
        (0, 0, 1, 0, 1, 0, 0),   # Z01
        (0, 1, 0, 1, 0, 0, 0),   # Z10
        (1, -1, 1, 0, 0, 1, 0),  # Z11
    ]

    col = 0
    for j_block in range(2):  # A col-block / B row-block: 0 or 1
        for k_block in range(4):  # B col-block: 0, 1, 2, 3
            for s in range(7):  # Strassen term
                # U part: which A elements are involved
                a00_c, a01_c, a10_c, a11_c, _, _, _, _ = strassen_terms[s]
                # A row 0, col 2*j_block: index 0*4 + 2*j_block
                if a00_c != 0:
                    U[0 * m + 2 * j_block][col] += a00_c
                # A row 0, col 2*j_block + 1: index 0*4 + 2*j_block + 1
                if a01_c != 0:
                    U[0 * m + 2 * j_block + 1][col] += a01_c
                # A row 1, col 2*j_block: index 1*4 + 2*j_block
                if a10_c != 0:
                    U[1 * m + 2 * j_block][col] += a10_c
                # A row 1, col 2*j_block + 1: index 1*4 + 2*j_block + 1
                if a11_c != 0:
                    U[1 * m + 2 * j_block + 1][col] += a11_c

                # V part: which B elements are involved
                _, _, _, _, b00_c, b01_c, b10_c, b11_c = strassen_terms[s]
                # B row 2*j_block, col 2*k_block: index 2*j_block*8 + 2*k_block
                if b00_c != 0:
                    V[(2 * j_block) * p + 2 * k_block][col] += b00_c
                # B row 2*j_block, col 2*k_block + 1: index 2*j_block*8 + 2*k_block + 1
                if b01_c != 0:
                    V[(2 * j_block) * p + 2 * k_block + 1][col] += b01_c
                # B row 2*j_block + 1, col 2*k_block: index (2*j_block+1)*8 + 2*k_block
                if b10_c != 0:
                    V[(2 * j_block + 1) * p + 2 * k_block][col] += b10_c
                # B row 2*j_block + 1, col 2*k_block + 1: index (2*j_block+1)*8 + 2*k_block + 1
                if b11_c != 0:
                    V[(2 * j_block + 1) * p + 2 * k_block + 1][col] += b11_c

                # W part: which C elements are involved
                for z in range(4):
                    zc = z_coeffs[z][s]
                    if zc != 0:
                        # z=0 => Z00: C row 0, col 2*k_block
                        # z=1 => Z01: C row 0, col 2*k_block + 1
                        # z=2 => Z10: C row 1, col 2*k_block
                        # z=3 => Z11: C row 1, col 2*k_block + 1
                        if z == 0:
                            c_row, c_col = 0, 2 * k_block
                        elif z == 1:
                            c_row, c_col = 0, 2 * k_block + 1
                        elif z == 2:
                            c_row, c_col = 1, 2 * k_block
                        else:
                            c_row, c_col = 1, 2 * k_block + 1
                        # W index: k*n + i = c_col * n + c_row
                        W[c_col * n + c_row][col] += zc

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
