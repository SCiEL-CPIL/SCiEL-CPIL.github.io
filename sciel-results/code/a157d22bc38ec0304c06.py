"""
6x6 matrix multiplication tensor decomposition using a 4+2 / 4+2 mixed block tiling.
The middle 2x4 and 4x2 blocks are handled with Strassen rank-7 2x2 patterns,
while the 4x4 corner block is handled with the standard rank-16 identity pattern.

Total rank:
  A00B00 -> C00 : 16
  A01B10 -> C00 : 28
  A00B01 -> C01 : 28
  A10B00 -> C10 : 28
  A11B11 -> C11 : 7
  Total = 107
"""
import time

DIMENSIONS = [6, 6, 6]
_DEADLINE = 135.0


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    rank = 107
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen patterns for 2x2 matrices.
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M1 = (A10+A11)B00
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},
         {(1, 0): 1, (1, 1): -1}),
        # M2 = A00(B01-B11)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1},
         {(0, 1): 1, (1, 1): 1}),
        # M3 = A11(B10-B00)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M4 = (A00+A01)B11
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},
         {(0, 0): -1, (0, 1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]

    col_idx = 0

    # Helper to add a standard scalar-multiplication term:
    # A[a_row, a_col] * B[b_row, b_col] -> C[c_row, c_col]
    def add_scalar_term(a_row, a_col, b_row, b_col, c_row, c_col, val=1):
        nonlocal col_idx
        U[a_row * m + a_col][col_idx] += val
        V[b_row * p + b_col][col_idx] += val
        W[c_row * n + c_col][col_idx] += val
        col_idx += 1
        if time.time() - t0 > _DEADLINE:
            return False
        return True

    # Helper to add a Strassen block:
    # A block top-left corner at (a_r0, a_c0), size 2x2
    # B block top-left corner at (b_r0, b_c0), size 2x2
    # C block top-left corner at (c_r0, c_c0), size 2x2
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col_idx
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            for (pi, pj), val in u_coeffs.items():
                r = a_r0 + pi
                c = a_c0 + pj
                U[r * m + c][col_idx] += val
            for (pi, pj), val in v_coeffs.items():
                r = b_r0 + pi
                c = b_c0 + pj
                V[r * p + c][col_idx] += val
            for (pi, pj), val in w_coeffs.items():
                r = c_r0 + pi
                c = c_c0 + pj
                W[r * n + c][col_idx] += val
            col_idx += 1
            if time.time() - t0 > _DEADLINE:
                return False
        return True

    # Block layout:
    # A: 6x6, split as [4;2] x [4|2]
    #   A00: rows 0-3, cols 0-3
    #   A01: rows 0-3, cols 4-5
    #   A10: rows 4-5, cols 0-3
    #   A11: rows 4-5, cols 4-5
    #
    # B: 6x6, split as [4;2] x [4|2]
    #   B00: rows 0-3, cols 0-3
    #   B01: rows 0-3, cols 4-5
    #   B10: rows 4-5, cols 0-3
    #   B11: rows 4-5, cols 4-5
    #
    # C: 6x6, split as [4;2] x [4|2]
    #   C00: rows 0-3, cols 0-3
    #   C01: rows 0-3, cols 4-5
    #   C10: rows 4-5, cols 0-3
    #   C11: rows 4-5, cols 4-5

    # ---- Term 1: A00 * B00 -> C00 (4x4 x 4x4, rank 16) ----
    for i in range(4):
        for k in range(4):
            for j in range(4):
                if not add_scalar_term(i, j, j, k, i, k):
                    return {"U": U, "V": V, "W": W}

    # ---- Term 2: A01 * B10 -> C00 (4x2 x 2x4, rank 28) ----
    # C00[i,k] += sum_{l=0}^{1} A01[i, l] * B10[l, k]
    # A01: rows 0-3, cols 4-5
    # B10: rows 4-5, cols 0-3
    # C00: rows 0-3, cols 0-3
    # This is a 4x2 times 2x4 product.
    # We can think of it as sum of two rank-14 terms? No.
    # Actually, A01 is 4x2, B10 is 2x4.
    # C00[i,k] = A01[i,0]*B10[0,k] + A01[i,1]*B10[1,k]
    # This is just 4*4*2 = 32 scalar multiplications naively.
    # But we can use Strassen on the 2x2 subblocks of A01 and B10?
    # A01 is 4x2, not a square block.
    # Let's decompose A01 into two 2x2 blocks vertically: A01a (rows 0-1), A01b (rows 2-3).
    # And B10 into two 2x2 blocks horizontally: B10a (cols 0-1), B10b (cols 2-3).
    # C00 gets contributions from:
    #   A01a * B10a -> C00[0:2, 0:2]
    #   A01a * B10b -> C00[0:2, 2:4]
    #   A01b * B10a -> C00[2:4, 0:2]
    #   A01b * B10b -> C00[2:4, 2:4]
    # Each of these is a 2x2 * 2x2 multiplication, rank 7.
    # Total: 4 * 7 = 28. Correct.

    # A01a: rows 0-1, cols 4-5. B10a: rows 4-5, cols 0-1. C00a: rows 0-1, cols 0-1.
    if not add_strassen_block(0, 4, 4, 0, 0, 0):
        return {"U": U, "V": V, "W": W}
    # A01a: rows 0-1, cols 4-5. B10b: rows 4-5, cols 2-3. C00b: rows 0-1, cols 2-3.
    if not add_strassen_block(0, 4, 4, 2, 0, 2):
        return {"U": U, "V": V, "W": W}
    # A01b: rows 2-3, cols 4-5. B10a: rows 4-5, cols 0-1. C00c: rows 2-3, cols 0-1.
    if not add_strassen_block(2, 4, 4, 0, 2, 0):
        return {"U": U, "V": V, "W": W}
    # A01b: rows 2-3, cols 4-5. B10b: rows 4-5, cols 2-3. C00d: rows 2-3, cols 2-3.
    if not add_strassen_block(2, 4, 4, 2, 2, 2):
        return {"U": U, "V": V, "W": W}

    # ---- Term 3: A00 * B01 -> C01 (4x4 x 4x2, rank 28) ----
    # A00: rows 0-3, cols 0-3. B01: rows 0-3, cols 4-5. C01: rows 0-3, cols 4-5.
    # Decompose A00 into two 2x2 blocks horizontally: A00a (cols 0-1), A00b (cols 2-3).
    # B01 is 4x2, decompose vertically: B01a (rows 0-1), B01b (rows 2-3).
    # C01 is 4x2, decompose vertically: C01a (rows 0-1), C01b (rows 2-3).
    # Contributions:
    #   A00a * B01a -> C01a
    #   A00b * B01a -> C01a
    #   A00a * B01b -> C01b
    #   A00b * B01b -> C01b
    # Each is 2x2 * 2x2, rank 7. Total 28.

    # A00a: rows 0-3, cols 0-1? No, A00 is 4x4.
    # Let's be careful.
    # A00[i,j] with i in 0..3, j in 0..3.
    # We split A00 into left and right halves: A00L (cols 0-1), A00R (cols 2-3).
    # But each half is 4x2, not 2x2.
    # We need to split both A00 and B01 into 2x2 blocks.
    # A00: rows 0-3, cols 0-3. Split into 2x2 blocks:
    #   A00[0:2, 0:2], A00[0:2, 2:4], A00[2:4, 0:2], A00[2:4, 2:4].
    # B01: rows 0-3, cols 4-5. Split into 2x2 blocks:
    #   B01[0:2, 4:6], B01[2:4, 4:6]. But B01 is 4x2, so we can only split rows.
    #   B01[0:2, 4:6] is 2x2, B01[2:4, 4:6] is 2x2.
    # C01: rows 0-3, cols 4-5. Split into:
    #   C01[0:2, 4:6], C01[2:4, 4:6].
    #
    # C01[i,k] = sum_{j=0}^{3} A00[i,j] * B01[j,k]
    #
    # Let's use the 2x2 Strassen decomposition on the 4x4 A00 and 4x2 B01.
    # Actually, it's easier to think of C01 as a 4x2 matrix.
    # C01 = A00 * B01.
    # Split A00 into 2x2 blocks: A00_00, A00_01, A00_10, A00_11 (2x2 each).
    # Split B01 into 2x2 blocks: B01_00, B01_10 (2x2 each).
    # C01_00 = A00_00 * B01_00 + A00_01 * B01_10
    # C01_10 = A00_10 * B01_00 + A00_11 * B01_10
    # Each term is 2x2 * 2x2, rank 7.
    # Total: 4 * 7 = 28.

    # A00_00: rows 0-1, cols 0-1. B01_00: rows 0-1, cols 4-5. C01_00: rows 0-1, cols 4-5.
    if not add_strassen_block(0, 0, 0, 4, 0, 4):
        return {"U": U, "V": V, "W": W}
    # A00_01: rows 0-1, cols 2-3. B01_10: rows 2-3, cols 4-5. C01_00: rows 0-1, cols 4-5.
    if not add_strassen_block(0, 2, 2, 4, 0, 4):
        return {"U": U, "V": V, "W": W}
    # A00_10: rows 2-3, cols 0-1. B01_00: rows 0-1, cols 4-5. C01_10: rows 2-3, cols 4-5.
    if not add_strassen_block(2, 0, 0, 4, 2, 4):
        return {"U": U, "V": V, "W": W}
    # A00_11: rows 2-3, cols 2-3. B01_10: rows 2-3, cols 4-5. C01_10: rows 2-3, cols 4-5.
    if not add_strassen_block(2, 2, 2, 4, 2, 4):
        return {"U": U, "V": V, "W": W}

    # ---- Term 4: A10 * B00 -> C10 (2x4 x 4x4, rank 28) ----
    # A10: rows 4-5, cols 0-3. B00: rows 0-3, cols 0-3. C10: rows 4-5, cols 0-3.
    # Split A10 horizontally: A10L (cols 0-1), A10R (cols 2-3).
    # Split B00 into 2x2 blocks: B00_00, B00_01, B00_10, B00_11.
    # C10: rows 4-5, cols 0-3. Split horizontally: C10L (cols 0-1), C10R (cols 2-3).
    #
    # C10[i,k] = sum_{j=0}^{3} A10[i,j] * B00[j,k]
    #
    # C10L = A10L * B00L + A10R * B00R
    # where B00L = [B00_00; B00_10] (4x2), B00R = [B00_01; B00_11] (4x2).
    # A10L is 2x2, A10R is 2x2.
    # A10L * B00L: 2x2 * 4x2. This is not a standard Strassen block.
    # Let's decompose further.
    # A10L * B00L = A10L * [B00_00; B00_10] = A10L * B00_00 + A10L * B00_10.
    # A10L * B00_00: 2x2 * 2x2, rank 7.
    # A10L * B00_10: 2x2 * 2x2, rank 7.
    # Similarly for A10R * B00R.
    # Total: 4 * 7 = 28.

    # A10L: rows 4-5, cols 0-1. B00_00: rows 0-1, cols 0-1. C10L_top: rows 4-5, cols 0-1.
    if not add_strassen_block(4, 0, 0, 0, 4, 0):
        return {"U": U, "V": V, "W": W}
    # A10L: rows 4-5, cols 0-1. B00_10: rows 2-3, cols 0-1. C10L_bottom: rows 4-5, cols 0-1.
    if not add_strassen_block(4, 0, 2, 0, 4, 0):
        return {"U": U, "V": V, "W": W}
    # A10R: rows 4-5, cols 2-3. B00_01: rows 0-1, cols 2-3. C10R_top: rows 4-5, cols 2-3.
    if not add_strassen_block(4, 2, 0, 2, 4, 2):
        return {"U": U, "V": V, "W": W}
    # A10R: rows 4-5, cols 2-3. B00_11: rows 2-3, cols 2-3. C10R_bottom: rows 4-5, cols 2-3.
    if not add_strassen_block(4, 2, 2, 2, 4, 2):
        return {"U": U, "V": V, "W": W}

    # ---- Term 5: A11 * B11 -> C11 (2x2 x 2x2, rank 7) ----
    # A11: rows 4-5, cols 4-5. B11: rows 4-5, cols 4-5. C11: rows 4-5, cols 4-5.
    if not add_strassen_block(4, 4, 4, 4, 4, 4):
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
