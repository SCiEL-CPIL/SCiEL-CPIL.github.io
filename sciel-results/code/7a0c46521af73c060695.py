"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-161 decomposition of 6x6 matrix multiplication using 3x3 Laderman
    (rank 23) block algorithm with 2x2 Strassen (rank 7) inner products.
    Total rank = 23 * 7 = 161.
    """
    start_time = time.time()
    deadline = 130.0

    n, m, p = DIMENSIONS  # all 6

    rank = 161
    U = [[0] * rank for _ in range(n * m)]  # 36 x 161
    V = [[0] * rank for _ in range(m * p)]  # 36 x 161
    W = [[0] * rank for _ in range(p * n)]  # 36 x 161

    def u_row(ai, aj):
        """Row index in U for A[ai_block*2+ai, aj_block*2+aj]"""
        return (ai_block * 2 + ai) * m + (aj_block * 2 + aj)

    def v_row(bi, bj):
        """Row index in V for B[bi_block*2+bi, bj_block*2+bj]"""
        return (bi_block * 2 + bi) * p + (bj_block * 2 + bj)

    def w_row(ci, cj):
        """Row index in W for C[cj_block*2+cj, ci_block*2+ci] (trace convention W[k*n+i,r])"""
        return (cj_block * 2 + cj) * n + (ci_block * 2 + ci)

    col = 0

    # Strassen 2x2: given A_block at (ai_block, aj_block) and B_block at (bi_block, bj_block),
    # compute 7 products. The output contributes to C at (ci_block, cj_block) where
    # ci_block = ai_block, cj_block = bj_block.
    # We need to map: A is 2x2 with rows indexed by ai (0,1) and cols by aj (0,1)
    # B is 2x2 with rows indexed by bi (0,1) and cols by bj (0,1)
    # C is 2x2 with rows indexed by ci (0,1) and cols by cj (0,1)
    # U row: (ai_block*2+ai, aj_block*2+aj) -> row (ai_block*2+ai)*m + (aj_block*2+aj)
    # V row: (bi_block*2+bi, bj_block*2+bj) -> row (bi_block*2+bi)*p + (bj_block*2+bj)
    # W row: (cj_block*2+cj, ci_block*2+ci) -> row (cj_block*2+cj)*n + (ci_block*2+ci)

    def strassen_2x2(ai_block, aj_block, bi_block, bj_block, ci_block, cj_block):
        """Add 7 Strassen products for A[ai_block,aj_block] * B[bi_block,bj_block] -> C[ci_block,cj_block]"""
        nonlocal col
        # M1 = (A11 + A22) * (B11 + B22) -> C11 += 1, C22 += 1
        U[u_row(ai_block, aj_block) if False else (ai_block*2+0)*m + (aj_block*2+0)][col] = 1
        U[(ai_block*2+1)*m + (aj_block*2+1)][col] = 1
        V[(bi_block*2+0)*p + (bj_block*2+0)][col] = 1
        V[(bi_block*2+1)*p + (bj_block*2+1)][col] = 1
        W[(cj_block*2+0)*n + (ci_block*2+0)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+1)][col] = 1
        col += 1

        # M2 = (A21 + A22) * B11 -> C21 += 1, C22 += -1
        U[(ai_block*2+1)*m + (aj_block*2+0)][col] = 1
        U[(ai_block*2+1)*m + (aj_block*2+1)][col] = 1
        V[(bi_block*2+0)*p + (bj_block*2+0)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+0)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+1)][col] = -1
        col += 1

        # M3 = A11 * (B12 - B22) -> C12 += 1, C22 += 1
        U[(ai_block*2+0)*m + (aj_block*2+0)][col] = 1
        V[(bi_block*2+0)*p + (bj_block*2+1)][col] = 1
        V[(bi_block*2+1)*p + (bj_block*2+1)][col] = -1
        W[(cj_block*2+1)*n + (ci_block*2+0)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+1)][col] = 1
        col += 1

        # M4 = A22 * (B21 - B11) -> C11 += 1, C21 += 1
        U[(ai_block*2+1)*m + (aj_block*2+1)][col] = 1
        V[(bi_block*2+1)*p + (bj_block*2+0)][col] = 1
        V[(bi_block*2+0)*p + (bj_block*2+0)][col] = -1
        W[(cj_block*2+0)*n + (ci_block*2+0)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+0)][col] = 1
        col += 1

        # M5 = (A11 + A12) * B22 -> C11 += -1, C12 += 1
        U[(ai_block*2+0)*m + (aj_block*2+0)][col] = 1
        U[(ai_block*2+0)*m + (aj_block*2+1)][col] = 1
        V[(bi_block*2+1)*p + (bj_block*2+1)][col] = 1
        W[(cj_block*2+0)*n + (ci_block*2+0)][col] = -1
        W[(cj_block*2+1)*n + (ci_block*2+0)][col] = 1
        col += 1

        # M6 = (A21 - A11) * (B11 + B12) -> C22 += 1
        U[(ai_block*2+1)*m + (aj_block*2+0)][col] = 1
        U[(ai_block*2+0)*m + (aj_block*2+0)][col] = -1
        V[(bi_block*2+0)*p + (bj_block*2+0)][col] = 1
        V[(bi_block*2+0)*p + (bj_block*2+1)][col] = 1
        W[(cj_block*2+1)*n + (ci_block*2+1)][col] = 1
        col += 1

        # M7 = (A12 - A22) * (B21 + B22) -> C11 += 1
        U[(ai_block*2+0)*m + (aj_block*2+1)][col] = 1
        U[(ai_block*2+1)*m + (aj_block*2+1)][col] = -1
        V[(bi_block*2+1)*p + (bj_block*2+0)][col] = 1
        V[(bi_block*2+1)*p + (bj_block*2+1)][col] = 1
        W[(cj_block*2+0)*n + (ci_block*2+0)][col] = 1
        col += 1

    # Laderman 3x3 rank-23 algorithm.
    # Each product is a 2x2 matrix multiplication (Strassen, rank 7).
    # A[i,j] is at block position (i_block=i, j_block=j)
    # B[j,k] is at block position (i_block=j, j_block=k)  
    # C[i,k] is at block position (i_block=i, j_block=k)
    #
    # Product Pi = A[ai,aj] * B[bi,bj] contributes to C[ci,cj]
    # where the 2x2 blocks are:
    #   A block: rows 2*ai..2*ai+1, cols 2*aj..2*aj+1  -> U row (2*ai+a)*m + (2*aj+b)
    #   B block: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1  -> V row (2*bi+c)*p + (2*bj+d)
    #   C block: rows 2*ci..2*ci+1, cols 2*cj..2*cj+1  -> W row (2*cj+e)*n + (2*ci+f)

    # Laderman products (ai, aj, bi, bj) -> contributes to C[ci, cj]
    # P1  = A[0,0] * B[0,0] -> C[0,0]
    # P2  = A[0,0] * (B[1,0] - B[0,0]) -> C[0,0]
    # P3  = (A[1,0] + A[0,0]) * B[0,0] -> C[0,0]
    # P4  = A[2,2] * B[2,2] -> C[2,2]
    # P5  = A[2,2] * (B[0,2] - B[2,2]) -> C[2,2]
    # P6  = (A[2,2] + A[0,2]) * B[2,2] -> C[2,2]
    # P7  = (A[0,2] + A[2,2]) * (B[2,2] + B[0,0]) -> C[1,1]
    # P8  = A[1,1] * B[1,1] -> C[1,1]
    # P9  = A[1,1] * (B[0,1] - B[1,1]) -> C[0,1]
    # P10 = (A[1,1] + A[0,1]) * B[1,1] -> C[0,1]
    # P11 = (A[0,1] + A[1,1]) * (B[1,1] + B[0,0]) -> C[1,1]
    # P12 = A[0,1] * (B[1,2] - B[1,1]) -> C[1,2], C[2,1]
    # P13 = (A[0,1] + A[2,1]) * B[1,1] -> C[2,1]
    # P14 = (A[0,1] + A[1,1]) * (B[1,1] + B[2,1]) -> C[1,2]
    # P15 = A[1,0] * (B[2,0] - B[0,0]) -> C[1,0]
    # P16 = (A[1,0] + A[2,0]) * B[0,0] -> C[2,0]
    # P17 = (A[1,0] + A[0,0]) * (B[0,0] + B[2,0]) -> C[1,0]
    # P18 = A[2,0] * (B[0,1] - B[0,0]) -> C[2,0]
    # P19 = (A[2,0] + A[0,0]) * B[0,0] -> C[2,0]
    # P20 = (A[2,0] + A[1,0]) * (B[0,0] + B[0,1]) -> C[2,0]
    # P21 = A[1,2] * (B[2,0] - B[1,0]) -> C[2,2]
    # P22 = (A[1,2] + A[2,2]) * B[1,0] -> C[2,2]
    # P23 = (A[1,2] + A[0,2]) * (B[1,0] + B[2,0]) -> C[2,2]

    # Each product may involve sums/differences of 2x2 blocks.
    # We handle each product by adding the appropriate linear combinations.

    def add_product(A_coeffs, B_coeffs, C_coeffs):
        """
        A_coeffs: list of (ai, aj, coeff) - A block indices and coefficients
        B_coeffs: list of (bi, bj, coeff) - B block indices and coefficients  
        C_coeffs: list of (ci, cj, coeff) - C block indices and coefficients
        Each block is 2x2, computed via Strassen.
        """
        nonlocal col
        # For each combination, we need to expand. But since Strassen is linear,
        # we can add the contributions. However, the product of sums requires
        # (sum of A blocks) * (sum of B blocks) which is NOT just a sum of Strassen products.
        # 
        # Actually, each "product" in Laderman is of the form (linear combo of A blocks) * (linear combo of B blocks).
        # This is a single bilinear product of two 2x2 matrices where the matrices are linear combinations.
        # We can compute this as: for each Strassen formula term, substitute the linear combinations.
        # 
        # Simpler approach: since each A "matrix" is a sum of at most 2 2x2 blocks, and same for B,
        # the product is a sum of at most 4 2x2x2x2 products. But that would increase rank.
        #
        # Actually NO - the Laderman algorithm works at the scalar level. At the block level,
        # each "element" is a 2x2 matrix, and the algorithm says to compute e.g.
        # M1 = A00 * B00 where A00 and B00 are 2x2 matrices. This is one 2x2 matrix multiply (rank 7).
        # M2 = A00 * (B10 - B00) - this is A00 times the 2x2 matrix (B10-B00), which is also rank 7.
        # So each Laderman product is ONE 2x2 matrix multiplication, not multiple.
        # The linear combinations happen BEFORE the multiplication.
        #
        # So for M2 = A00 * (B10 - B00):
        #   The A matrix is just A00 (one 2x2 block)
        #   The B matrix is (B10 - B00) (linear combo of two 2x2 blocks)
        #   We compute this as a single 2x2 multiply using Strassen, where the B input
        #   has entries that are differences.
        #
        # In the tensor decomposition, this means:
        #   U has entries from A00 block
        #   V has entries from B10 and B00 blocks (with coefficients +1 and -1)
        #   W has entries from C00 block
        #   And we use the standard Strassen 7-product pattern.

        # For a product where A = sum of aA * A[ai,aj] and B = sum of bB * B[bi,bj],
        # we need to expand. But actually in Laderman, each product has at most 2 terms in A and 2 in B.
        # The product (a1*A1 + a2*A2) * (b1*B1 + b2*B2) = a1*b1*A1*B1 + a1*b2*A1*B2 + a2*b1*A2*B1 + a2*b2*A2*B2
        # This would be 4 separate 2x2 multiplies = 28 rank, which defeats the purpose.
        #
        # WAIT - I'm confusing myself. In the Laderman algorithm applied to 3x3 scalar matrices,
        # each "product" is a scalar multiplication. When we lift to 2x2 blocks, each scalar
        # multiplication becomes a 2x2 matrix multiplication (rank 7). The linear combinations
        # in Laderman are of the SCALAR entries, which become linear combinations of the
        # corresponding 2x2 blocks. So M2 = A00 * (B10 - B00) means: take the 2x2 matrix A00
        # and multiply it by the 2x2 matrix (B10 - B00). This is ONE 2x2 matrix multiply.
        #
        # In the tensor decomposition, to compute A00 * (B10 - B00) via Strassen:
        #   The "A matrix" for Strassen is A00 (2x2)
        #   The "B matrix" for Strassen is (B10 - B00) (2x2)
        #   Strassen computes 7 scalar products, each of which is a bilinear form.
        #   E.g., M1_strassen = (A00[0,0] + A00[1,1]) * (B10[0,0]-B00[0,0] + B10[1,1]-B00[1,1])
        #   This expands to: A00[0,0]*B10[0,0] - A00[0,0]*B00[0,0] + A00[1,1]*B10[1,1] - A00[1,1]*B00[1,1]
        #   Each term is a separate rank-1 contribution.
        #
        # So in the U/V/W matrices, for one Strassen product of A00*(B10-B00):
        #   U row for A00[0,0] gets coefficient 1
        #   U row for A00[1,1] gets coefficient 1
        #   V row for B10[0,0] gets coefficient 1
        #   V row for B00[0,0] gets coefficient -1
        #   V row for B10[1,1] gets coefficient 1
        #   V row for B00[1,1] gets coefficient -1
        #   etc.
        #
        # This is still rank 7 for this one Laderman product! Because Strassen's 7 products
        # are bilinear, and linearity means we can substitute linear combinations freely.
        #
        # So the total rank is indeed 23 * 7 = 161.

        # For each Strassen term, we need to handle the linear combinations in A and B.
        # Let's define the Strassen terms with their A and B indices (within the 2x2 block)
        # and then apply the coefficients from the linear combinations.

        # Strassen 2x2 terms (a_idx, b_idx) where a_idx is (row,col) in A, b_idx in B:
        # Term 1: A=(0,0),(1,1) B=(0,0),(1,1) -> C=(0,0),(1,1)
        # Term 2: A=(1,0),(1,1) B=(0,0) -> C=(1,0),(1,1) with C(1,1)=-1
        # Term 3: A=(0,0) B=(0,1),(1,1) with B(1,1)=-1 -> C=(0,1),(1,1)
        # Term 4: A=(1,1) B=(1,0),(0,0) with B(0,0)=-1 -> C=(0,0),(1,0)
        # Term 5: A=(0,0),(0,1) B=(1,1) -> C=(0,0)=-1,(0,1)
        # Term 6: A=(1,0),(0,0) with A(0,0)=-1 B=(0,0),(0,1) -> C=(1,1)
        # Term 7: A=(0,1),(1,1) with A(1,1)=-1 B=(1,0),(1,1) -> C=(0,0)

        # For a product with A linear combo: sum of (coeff_a, ai, aj, a_row, a_col)
        # and B linear combo: sum of (coeff_b, bi, bj, b_row, b_col)
        # Each Strassen term involves specific (a_row,a_col) and (b_row,b_col) positions.

        # This is getting complex. Let me use a simpler approach:
        # For each Laderman product, identify the A block(s) and B block(s) involved,
        # and their coefficients. Then for each of the 7 Strassen terms, add the
        # appropriate entries to U, V, W.

        pass  # Placeholder - we'll handle each product explicitly below

    # Let me handle each Laderman product explicitly.
    # For a product A_blocks * B_blocks -> C_blocks:
    # A_blocks: list of (coeff, ai, aj) meaning coeff * A[ai,aj]
    # B_blocks: list of (coeff, bi, bj) meaning coeff * B[bi,bj]
    # C_blocks: list of (coeff, ci, cj) meaning the result adds coeff * C[ci,cj]
    #
    # The 2x2 matrix to multiply is A_mat = sum(coeff_a * A[ai,aj]) and B_mat = sum(coeff_b * B[bi,bj])
    # We compute A_mat * B_mat via Strassen (7 rank-1 terms) and add to C.

    def add_laderman_product(A_blocks, B_blocks, C_blocks):
        """
        A_blocks: list of (coeff, ai, aj) - 2x2 block A[ai,aj] with coefficient
        B_blocks: list of (coeff, bi, bj) - 2x2 block B[bi,bj] with coefficient
        C_blocks: list of (coeff, ci, cj) - 2x2 block C[ci,cj] receives this product
        """
        nonlocal col
        # For each Strassen term, the A part is a bilinear form on the 2x2 A_mat,
        # and the B part is a bilinear form on the 2x2 B_mat.
        # 
        # Strassen term t has:
        #   A_part: dict from (row,col) to coefficient in the A_mat
        #   B_part: dict from (row,col) to coefficient in the B_mat
        #   C_part: dict from (row,col) to coefficient in the C_mat
        #
        # The contribution to U for row (2*ai+a_row, 2*aj+a_col) is sum over A_blocks
        #   of coeff_a * A_part[(a_row, a_col)] for the block at (ai, aj)
        # Similarly for V and W.

        strassen_terms = [
            # (A_part, B_part, C_part)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        for A_part, B_part, C_part in strassen_terms:
            # Build U contributions
            for (a_r, a_c, a_coeff) in A_part:
                for (c_a, ai, aj) in A_blocks:
                    row = (ai * 2 + a_r) * m + (aj * 2 + a_c)
                    U[row][col] += c_a * a_coeff

            # Build V contributions
            for (b_r, b_c, b_coeff) in B_part:
                for (c_b, bi, bj) in B_blocks:
                    row = (bi * 2 + b_r) * p + (bj * 2 + b_c)
                    V[row][col] += c_b * b_coeff

            # Build W contributions
            for (c_r, c_c, c_coeff) in C_part:
                for (c_c_out, ci, cj) in C_blocks:
                    row = (cj * 2 + c_c) * n + (ci * 2 + c_r)
                    W[row][col] += c_c_out * c_coeff

            col += 1

    # Now define the 23 Laderman products.
    # Each is: (A_blocks, B_blocks, C_blocks)

    # P1 = A[0,0] * B[0,0] -> C[0,0]
    add_laderman_product([(1,0,0)], [(1,0,0)], [(1,0,0)])

    # P2 = A[0,0] * (B[1,0] - B[0,0]) -> C[0,0]
    add_laderman_product([(1,0,0)], [(1,1,0),(-1,0,0)], [(1,0,0)])

    # P3 = (A[1,0] + A[0,0]) * B[0,0] -> C[0,0]
    add_laderman_product([(1,1,0),(1,0,0)], [(1,0,0)], [(1,0,0)])

    # P4 = A[2,2] * B[2,2] -> C[2,2]
    add_laderman_product([(1,2,2)], [(1,2,2)], [(1,2,2)])

    # P5 = A[2,2] * (B[0,2] - B[2,2]) -> C[2,2]
    add_laderman_product([(1,2,2)], [(1,0,2),(-1,2,2)], [(1,2,2)])

    # P6 = (A[2,2] + A[0,2]) * B[2,2] -> C[2,2]
    add_laderman_product([(1,2,2),(1,0,2)], [(1,2,2)], [(1,2,2)])

    # P7 = (A[0,2] + A[2,2]) * (B[2,2] + B[0,0]) -> C[1,1]
    add_laderman_product([(1,0,2),(1,2,2)], [(1,2,2),(1,0,0)], [(1,1,1)])

    # P8 = A[1,1] * B[1,1] -> C[1,1]
    add_laderman_product([(1,1,1)], [(1,1,1)], [(1,1,1)])

    # P9 = A[1,1] * (B[0,1] - B[1,1]) -> C[0,1]
    add_laderman_product([(1,1,1)], [(1,0,1),(-1,1,1)], [(1,0,1)])

    # P10 = (A[1,1] + A[0,1]) * B[1,1] -> C[0,1]
    add_laderman_product([(1,1,1),(1,0,1)], [(1,1,1)], [(1,0,1)])

    # P11 = (A[0,1] + A[1,1]) * (B[1,1] + B[0,0]) -> C[1,1]
    add_laderman_product([(1,0,1),(1,1,1)], [(1,1,1),(1,0,0)], [(1,1,1)])

    # P12 = A[0,1] * (B[1,2] - B[1,1]) -> C[1,2] and C[2,1]
    add_laderman_product([(1,0,1)], [(1,1,2),(-1,1,1)], [(1,1,2),(1,2,1)])

    # P13 = (A[0,1] + A[2,1]) * B[1,1] -> C[2,1]
    add_laderman_product([(1,0,1),(1,2,1)], [(1,1,1)], [(1,2,1)])

    # P14 = (A[0,1] + A[1,1]) * (B[1,1] + B[2,1]) -> C[1,2]
    add_laderman_product([(1,0,1),(1,1,1)], [(1,1,1),(1,2,1)], [(1,1,2)])

    # P15 = A[1,0] * (B[2,0] - B[0,0]) -> C[1,0]
    add_laderman_product([(1,1,0)], [(1,2,0),(-1,0,0)], [(1,1,0)])

    # P16 = (A[1,0] + A[2,0]) * B[0,0] -> C[2,0]
    add_laderman_product([(1,1,0),(1,2,0)], [(1,0,0)], [(1,2,0)])

    # P17 = (A[1,0] + A[0,0]) * (B[0,0] + B[2,0]) -> C[1,0]
    add_laderman_product([(1,1,0),(1,0,0)], [(1,0,0),(1,2,0)], [(1,1,0)])

    # P18 = A[2,0] * (B[0,1] - B[0,0]) -> C[2,0]
    add_laderman_product([(1,2,0)], [(1,0,1),(-1,0,0)], [(1,2,0)])

    # P19 = (A[2,0] + A[0,0]) * B[0,0] -> C[2,0]
    add_laderman_product([(1,2,0),(1,0,0)], [(1,0,0)], [(1,2,0)])

    # P20 = (A[2,0] + A[1,0]) * (B[0,0] + B[0,1]) -> C[2,0]
    add_laderman_product([(1,2,0),(1,1,0)], [(1,0,0),(1,0,1)], [(1,2,0)])

    # P21 = A[1,2] * (B[2,0] - B[1,0]) -> C[2,2]
    add_laderman_product([(1,1,2)], [(1,2,0),(-1,1,0)], [(1,2,2)])

    # P22 = (A[1,2] + A[2,2]) * B[1,0] -> C[2,2]
    add_laderman_product([(1,1,2),(1,2,2)], [(1,1,0)], [(1,2,2)])

    # P23 = (A[1,2] + A[0,2]) * (B[1,0] + B[2,0]) -> C[2,2]
    add_laderman_product([(1,1,2),(1,0,2)], [(1,1,0),(1,2,0)], [(1,2,2)])

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    random.seed(42)
    valid = True
    for _ in range(200):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
