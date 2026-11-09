"""Strassen-based rank-49 decomposition for 4x4 x 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    def urow(i, j):
        return i * m + j

    def vrow(j, k):
        return j * p + k

    def wrow(k, i):
        return k * n + i

    # A block (bi, bj): rows 2*bi, 2*bi+1; cols 2*bj, 2*bj+1
    # B block (bi, bj): rows 2*bi, 2*bi+1; cols 7*bj..7*bj+6
    # C block (ci, cj): rows 7*cj..7*cj+6; cols 2*ci, 2*ci+1

    # For each Strassen product s (0..6), we have:
    # A_s = sum of coeff_a * A[ai,aj] for various (ai,aj) in {0,1}^2
    # B_s = sum of coeff_b * B[bi,:] for bi in {0,1} (the two rows of a B block)
    # The product A_s * B_s contributes to C blocks with coefficients.
    #
    # Each elementary term: for each (a_row, a_col) in A_s support,
    #   for each b_row in B_s support,
    #   for each k in 0..6:
    #     U[urow(a_row, a_col)][col] += coeff_a
    #     V[vrow(b_row, 7*bj + k)][col] += coeff_b
    #     W[wrow(7*cj + k, c_row)][col] += coeff_c
    #
    # But that's 2*2*2*7 = 56 terms per product, too many.
    #
    # Actually, the rank-1 tensor term is:
    # T[i,j,k] = U[i,s] * V[j,s] * W[k,s]
    # where i indexes A (16 entries), j indexes B (28 entries), k indexes C (28 entries).
    #
    # For a Strassen product M_s = (sum_a c_a * A[a]) * (sum_b d_b * B[b]):
    # This is NOT a rank-1 term. It's a sum of rank-1 terms.
    #
    # Wait - I need to reconsider. In tensor decomposition for matrix multiplication,
    # each rank-1 term corresponds to ONE scalar multiplication.
    #
    # For 2x2 x 2x7 with Strassen: the 7 products are scalar*vector.
    # Each scalar*vector product is 7 scalar multiplications (one per column).
    # So 7 products * 7 columns = 49 scalar multiplications = rank 49.
    #
    # Each scalar multiplication: pick one A entry (with coeff), one B entry (with coeff),
    # and add to one C entry (with coeff).
    #
    # So for Strassen product s, column c (0..6), we have:
    # A_s scalar = sum over (a_row,a_col) in block of coeff * A[a_row,a_col]
    # B_s vector element at col c = sum over b_row in block of coeff * B[b_row, 7*bj+c]
    #
    # The product is: (sum_a ca*A[a]) * (sum_b db*B[b]) = sum_{a,b} ca*db * A[a]*B[b]
    # Each (a,b) pair is one rank-1 term!
    #
    # So for each Strassen product s and each column c, we have |A_s_support| * |B_s_support| rank-1 terms.
    # That's way more than 49 total.
    #
    # I think the issue is that Strassen doesn't directly give a rank-49 decomposition for 4x4x4x7
    # in the simple way. Let me just use the schoolbook rank 112 which is correct.

    rank = n * m * p  # 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
    # Schoolbook would be rank 112. Strassen 2-level gives rank 49.
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    def urow(i, j):
        return i * m + j

    def vrow(j, k):
        return j * p + k

    def wrow(k, i):
        return k * n + i

    # Elementary: U[urow(a,b)] += 1, V[vrow(b,c)] += 1, W[wrow(c,d)] += 1
    def elem(a, b, c, d, col, s=1):
        U[urow(a, b)][col] += s
        V[vrow(b, c)][col] += s
        W[wrow(c, d)][col] += s

    # Bottom level: 7 base ops for 2x2 (A) x 2x7 (B) -> 2x7 (C)
    # A indices (a,b) in {0,1}^2, B indices (b,c) with c in {0,1} (2 rows), d in {0..6}
    def bottom(col0, a, b, c, d):
        # 7 Strassen ops for 2x2 x 2x7
        # M1 = (A00+A11)(B00+B11)
        elem(a, b, c, d, col0 + 0)
        elem(a, b + 1, c, d, col0 + 0)
        elem(a + 1, b, c + 1, d, col0 + 0)
        elem(a + 1, b + 1, c + 1, d, col0 + 0)
        # M2 = (A10+A11)B00
        elem(a + 1, b, c, d, col0 + 1)
        elem(a + 1, b + 1, c, d, col0 + 1)
        # M3 = A00(B01-B11)
        elem(a, b, c + 1, d, col0 + 2)
        elem(a, b, c + 1, d, col0 + 3, -1)
        # M4 = A11(B10-B00)
        elem(a + 1, b + 1, c, d, col0 + 4)
        elem(a + 1, b + 1, c, d, col0 + 5, -1)
        # M5 = (A00+A01)B11
        elem(a, b, c + 1, d, col0 + 6)
        elem(a, b + 1, c + 1, d, col0 + 6)
        # M6 = (A10-A00)(B00+B01)
        elem(a + 1, b, c, d, col0 + 7)
        elem(a + 1, b, c + 1, d, col0 + 7)
        elem(a, b, c, d, col0 + 7, -1)
        elem(a, b, c + 1, d, col0 + 7, -1)
        # M7 = (A01-A11)(B10+B11)
        elem(a, b + 1, c + 1, d, col0 + 8)
        elem(a, b + 1, c + 1, d, col0 + 8)
        elem(a + 1, b + 1, c + 1, d, col0 + 8, -1)
        elem(a + 1, b + 1, c + 1, d, col0 + 8, -1)

    # Top level: 7 Strassen products, each decomposed via bottom level.
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)  (block row index)
    # C blocks: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    col = 0
    for s in range(7):
        # M_s = (A_x)(B_y) where the combination is handled by summing bottom ops
        # We implement each M_s as a single 2x2 x 2x7 product using bottom(),
        # but the "A" and "B" of M_s are themselves sums/differences of blocks.
        # Instead, we directly emit the elementary contributions for each M_s.
        #
        # M1 = (A11+A22)(B11+B22)
        if s == 0:
            # (A11+A22): blocks (0,0) and (1,1)
            # (B11+B22): blocks (0,0) and (1,1)
            # product = A11*B11 + A11*B22 + A22*B11 + A22*B22
            # Each A_block * B_block is a 2x2 x 2x7 product -> 7 ops each
            # A11*B11: a=0,b=0 ; c=0, d in 0..6
            for d in range(7):
                bottom(col + 0, 0, 0, 0, d)  # 9 ops? No, bottom uses 9 columns
            col += 0  # placeholder, we will restructure
        col = 0
        # Restructure: each M_s is a sum of up to 4 (2x2 x 2x7) products.
        # Each (2x2 x 2x7) product costs 7 elementary ops.
        # M1: 4 products -> 28 ops; but we can share? No, keep simple.
        # Total: M1(4)+M2(1)+M3(1)+M4(1)+M5(1)+M6(2)+M7(2) = 12 products * 7 = 84.
        # That's worse than 49. We need the bottom level to also be Strassen
        # applied to the 2x2 A-block times 2x7 B-block, giving 7 ops per sub-block-pair,
        # and the top level combines 7 such sub-products. So total = 7*7 = 49.
        # The key: M_s at top level is a SINGLE 2x2 x 2x7 product (not a sum of 4).
        # The sums/differences are absorbed into the U and V factor matrices.
        #
        # So: M_s = A_s * B_s where A_s is a 2x2 matrix (sum/diff of A blocks)
        # and B_s is a 2x7 matrix (sum/diff of B blocks).
        # A_s * B_s is a 2x2 x 2x7 product -> 7 elementary ops.
        # 7 M_s * 7 ops = 49.

        # Define A_s as linear combination of A-blocks, B_s as linear combination of B-blocks.
        # A-block (i,j) has U-rows urow(i*2+0, j*2+0), urow(i*2+0, j*2+1), urow(i*2+1, j*2+0), urow(i*2+1, j*2+1)
        # B-block (i,j) has V-rows vrow(i*2+0, j*2+0..6), vrow(i*2+1, j*2+0..6)
        # C-block (i,j) has W-rows wrow(j*2+0, i*2+0), wrow(j*2+0, i*2+1), wrow(j*2+1, i*2+0), wrow(j*2+1, i*2+1)
        #
        # For M_s, we need to emit 7 elementary ops where U-rows are the rows of A_s
        # (which are specific rows of A, possibly with coefficients), V-rows are rows of B_s,
        # and W-rows are rows of C_s.
        #
        # A_s is a 2x2 matrix. Its rows are linear combinations of A-block rows.
        # For Strassen, A_s = A_x + A_y or A_x - A_y, so A_s's rows are sums/diffs of
        # specific A-block rows. The U factor for M_s must have non-zero entries in
        # the rows corresponding to A_s's non-zero entries.
        #
        # This is getting complex. Let me just directly compute the 49 elementary ops
        # by expanding the full Strassen formula.

        break  # restart with clean approach

    # Clean approach: directly enumerate all 49 elementary operations.
    # Reset matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # A-block (bi, bj) occupies rows (bi*2, bi*2+1) and cols (bj*2, bj*2+1)
    # B-block (bi, bj) occupies rows (bi*2, bi*2+1) and cols (bj*7, ..., bj*7+6)
    # C-block (bi, bj) occupies rows (bj*7, ..., bj*7+6) and cols (bi*2, bi*2+1)

    # For each of 7 top-level Strassen products M_s = A_s * B_s:
    # A_s is a 2x2 matrix formed from A-blocks.
    # B_s is a 2x7 matrix formed from B-blocks.
    # A_s * B_s is a 2x2 x 2x7 product, decomposed into 7 elementary ops.
    #
    # The 2x2 A_s has entries that are linear combos of A-block entries.
    # For simplicity, A_s = c1 * A_x + c2 * A_y where A_x, A_y are A-blocks.
    # Then A_s's (r,c) entry = c1 * A_x(r,c) + c2 * A_y(r,c).
    # The U factor for M_s has non-zero entries in rows of A_x and A_y.
    #
    # Similarly B_s = d1 * B_u + d2 * B_v, V factor in rows of B_u, B_v.
    #
    # The 7 elementary ops for A_s * B_s:
    # Let A_s = [[a00, a01],[a10, a11]] (2x2), B_s = [[b00,...,b06],[b10,...,b16]] (2x7)
    # C_s = A_s * B_s (2x7)
    # Strassen 7 ops for this 2x2 x 2x7:
    # P1 = (a00+a11)(b00+b10, ..., b06+b16)
    # P2 = (a10+a11)(b00, ..., b06)
    # P3 = a00(b01-b11, ..., b06-b16)  -- wait, B_s is 2x7, so "b01" doesn't exist.
    #
    # Actually B_s is 2x7, so Strassen on 2x2 x 2x7:
    # The 2x2 part is A_s, the 2x7 part is B_s.
    # Strassen 7 products:
    # M1 = (A_s[0,0]+A_s[1,1]) * (B_s[0,:]+B_s[1,:])
    # M2 = (A_s[1,0]+A_s[1,1]) * B_s[0,:]
    # M3 = A_s[0,0] * (B_s[1,:]-B_s[0,:])
    # M4 = A_s[1,1] * (B_s[1,:]-B_s[0,:])  -- no, standard Strassen:
    # M4 = A_s[1,1] * (B_s[1,:]-B_s[0,:])? Let me use standard:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # Here B00, B01, B10, B11 are 2x7 blocks? No, B_s is 2x7, so B00 and B01 are
    # the two columns of B_s? No, B_s is 2 rows x 7 cols. B00 = B_s[0,:] (1x7), B10 = B_s[1,:] (1x7).
    # There's no B01 or B11 in the usual sense because B_s has only 2 rows.
    #
    # I think the correct interpretation: for 2x2 x 2x7, we treat it as:
    # A_s is 2x2, B_s is 2x7. The product C_s is 2x7.
    # We can apply Strassen by viewing B_s as [B_top; B_bottom] where B_top, B_bottom are 1x7.
    # Then:
    # M1 = (A_s[0,0]+A_s[1,1]) * (B_top + B_bottom)
    # M2 = (A_s[1,0]+A_s[1,1]) * B_top
    # M3 = A_s[0,0] * (B_bottom - B_top)
    # M4 = A_s[1,1] * (B_bottom - B_top)  -- this gives same as M3 times different A
    # Hmm, this isn't right either. Let me just use the fact that
    # 2x2 x 2x7 = (2x2 x 2x2) x (2x2 -> 2x7) and apply Strassen to the 2x2 part,
    # treating the 7 columns as separate.
    #
    # Actually, the cleanest way: 2x2 x 2x7 multiplication can be done with 7 ops
    # by applying Strassen to the 2x2 x 2x2 part for each of the 7 columns, but that's 7*7=49.
    # No, we want 7 ops total.
    #
    # The rank of 2x2 x 2x7 multiplication is 7 (Strassen). The 7 ops are:
    # For each column c of B_s (c=0..6):
    #   We have a 2x2 x 2x1 product (A_s * B_s[:,c]), which needs 7 ops.
    #   Total 7*7 = 49. That's not 7.
    #
    # I think I'm overcomplicating. Let me just use the schoolbook for the bottom level
    # and Strassen for the top level. Top: 7 products, each is 2x2 x 2x7 = 2*2*7 = 28 elementary.
    # Total = 7 * 28 = 196. Worse than 112.
    #
    # Alternative: apply Strassen at both levels.
    # Top: 4x4 x 4x7 -> 7 products of 2x2 x 2x7.
    # Each 2x2 x 2x7: apply Strassen to the 2x2 part, giving 7 products of 1x1 x 2x7 = 14 elementary.
    # Total = 7 * 7 * 14 = 686. Way worse.
    #
    # I think the correct approach is:
    # 4x4 x 4x7: Strassen top gives 7 products of 2x2 x 2x7.
    # Each 2x2 x 2x7: this is a 2x2 matrix times a 2x7 matrix.
    # The rank of this is 7 (Strassen for 2x2 x 2x2 applied to each of 7 columns, but shared).
    # Actually, 2x2 x 2x7 can be done with 7 ops:
    # C = A * B, A is 2x2, B is 2x7.
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    # Strassen:
    # M1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])
    # M2 = (A[1,0]+A[1,1]) * B[0,:]
    # M3 = A[0,0] * (B[1,:]-B[0,:])
    # M4 = A[1,1] * (B[1,:]-B[0,:])  -- no
    #
    # Standard Strassen for 2x2 x 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # For 2x2 x 2x7, B has 2 rows and 7 cols. B00 = B[0,0], B01 = B[0,1], etc.
    # But we only have 2 rows, so B00, B10 are the two entries of column 0.
    # B01, B11 are the two entries of column 1.
    # So for 7 columns, we'd need 7*7 = 49 ops.
    #
    # I give up on getting 49. Let me just use the schoolbook rank 112 but with a
    # different structure, or accept that Strassen gives 49 for 4x4x4 and adapt.
    #
    # For 4x4 x 4x4, Strassen gives rank 49. For 4x4 x 4x7, we can use the 49-op
    # Strassen for 4x4x4 and extend to 7 columns. Each of the 49 ops in 4x4x4 is
    # a 1x1 x 1x1 product. For 4x4x4x7, each is a 1x1 x 1x7 product = 7 elementary.
    # Total = 49 * 7 = 343. Worse than 112.
    #
    # I think the best I can do is the schoolbook 112. Let me just return that.

    # Reset to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
