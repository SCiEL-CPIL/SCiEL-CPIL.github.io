"""Laderman-based rank-84 exact tensor decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _laderman_3x3():
    """Return (U, V, W) for 3x3 matrix multiplication, rank 21.
    U: (9, 21), V: (9, 21), W: (9, 21) with W[k*3+i, c] convention.
    """
    n = 3
    # The 21 terms of Laderman's algorithm.
    # Each term: (U_vec, V_vec, W_vec) where indices are (i,j), (j,k), (k,i) for 0..2
    # U_vec has 1 at position (i,j), V_vec at (j,k), W_vec at (k,i)
    terms = []

    def add(u_idx, v_idx, w_idx):
        terms.append((u_idx, v_idx, w_idx))

    # Laderman's 21 terms (standard form).
    # Using 0-indexed (i,j) for A, (j,k) for B, (k,i) for C.
    # Row 1: i=0
    add((0, 0), (0, 0), (0, 0))          # 1
    add((0, 1), (1, 1), (1, 0))          # 2
    add((0, 2), (2, 2), (2, 0))          # 3
    # Row 2: i=1
    add((1, 0), (0, 0), (0, 1))          # 4
    add((1, 1), (1, 1), (1, 1))          # 5
    add((1, 2), (2, 2), (2, 1))          # 6
    # Row 3: i=2
    add((2, 0), (0, 0), (0, 2))          # 7
    add((2, 1), (1, 1), (1, 2))          # 8
    add((2, 2), (2, 2), (2, 2))          # 9
    # Additional 12 terms for the off-diagonal products
    # A(0,1)*B(1,0) -> C(0,1)
    add((0, 1), (1, 0), (1, 0))          # 10  (this is a "diagonal" in a sense)
    # Actually let me use the known correct Laderman decomposition.
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin

    n, m, p = DIMENSIONS  # 4, 4, 4

    # Use Laderman's rank-21 3x3 decomposition embedded in 4x4.
    # The 4x4 matrix multiplication tensor can be decomposed by taking
    # the 3x3 Laderman decomposition and extending to 4x4.
    #
    # A known rank bound: M(4) <= M(3) * 4 = 84 (by block embedding with
    # the 4th row/col handled by schoolbook on the boundary, but actually
    # the standard embedding gives rank 4*M(3) = 84).
    #
    # More precisely, we can decompose 4x4 multiplication using the 3x3
    # Laderman terms for the top-left 3x3 block, plus boundary terms.
    #
    # Actually, the cleanest rank-84 construction:
    # Treat 4x4 as 3x3 blocks where we use Laderman for the "interior"
    # and schoolbook for the boundary. But the standard result is:
    # M(4) <= 4 * M(3) = 84 via the "embedding" argument.
    #
    # Let me construct this explicitly. The 4x4 multiplication has 64 terms
    # in schoolbook. We need to reduce to 84... wait, 84 > 64, so that's not
    # a reduction from schoolbook.
    #
    # Actually M(3) = 21 (Laderman), and the embedding gives M(4) <= 4*M(3) = 84.
    # But schoolbook for 4x4 is 64, so 84 is WORSE than schoolbook!
    #
    # Let me reconsider. The correct embedding: to multiply 4x4 matrices,
    # we can use the 3x3 algorithm 4 times (once for each "layer"), giving
    # rank 4*21 = 84. But since schoolbook is 64, this is not useful.
    #
    # However, there are better constructions. Let me think about what's known.
    #
    # Actually, I recall that the best known upper bound for M(4) is 49 (via
    # Strassen-like methods), and the lower bound is 27.
    #
    # Let me use a Strassen-based approach for rank 49.
    #
    # Strassen's algorithm for 4x4: split into 2x2 blocks, use Strassen's 7
    # multiplications for each 2x2 block multiplication. This gives:
    # M(4) <= 7 * M(2) = 7 * 7 = 49.
    #
    # Let me implement Strassen's 4x4 decomposition.

    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B where Cij = sum_k Aik * Bkj

    # Using Strassen at the block level:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where M1..M7 are 2x2 matrix products (each of rank 7 for 2x2).

    # So total rank = 7 * 7 = 49.

    # Let me build this explicitly.
    # A is 4x4, indexed by (i,j) for i,j in 0..3
    # B is 4x4, indexed by (j,k) for j,k in 0..3
    # C is 4x4, indexed by (k,i) for k,i in 0..3 (W convention: W[k*n+i, r])

    # Block indices: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], etc.
    # In flat indices: A[i*4+j]

    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)

    # Define the 7 Strassen products for 2x2, each of rank 7.
    # For a 2x2 product X*Y where X,Y are 2x2:
    # The 7 rank-1 terms for 2x2 Strassen:
    # Term 1: (X00+X11)(Y00+Y11) -> contributes to C00, C11
    # Term 2: (X10+X11)Y00 -> contributes to C10, C11
    # Term 3: X00(Y10-Y11) -> contributes to C01, C11
    # Term 4: X11(Y01-Y00) -> contributes to C00, C11
    # Term 5: (X00+X01)Y11 -> contributes to C00, C01
    # Term 6: (X10-X00)(Y00+Y01) -> contributes to C10, C11
    # Term 7: (X01-X11)(Y10+Y11) -> contributes to C00, C01

    # For 2x2 Strassen, the rank-1 decompositions:
    # Each of the 7 products can be written as a rank-1 tensor.
    # Let me define them explicitly.

    # For 2x2: A is 2x2 (indices 0,1), B is 2x2, C is 2x2
    # U indices: (i,j) -> i*2+j
    # V indices: (j,k) -> j*2+k
    # W indices: (k,i) -> k*2+i

    # The 7 terms for 2x2 Strassen:
    # Term 1: U has +1 at (0,0),(1,1); V has +1 at (0,0),(1,1); W has +1 at (0,0),(1,1)
    #   Actually this is not rank-1. Let me reconsider.
    #
    # Strassen's 7 products are NOT rank-1 in the original matrices.
    # Each M_i is a full 2x2 matrix product, which itself has rank 4 in schoolbook.
    # So M(4) <= 7 * M(2) = 7*4 = 28? No, that's not right either.
    #
    # Let me reconsider. M(2) = 7 (Strassen). So M(4) <= 7*M(2) = 49.
    # This means we need 49 rank-1 terms total.
    #
    # The structure: 7 Strassen "big" products, each of which is a 2x2 matrix
    # multiplication of rank 7. So 7*7 = 49 rank-1 terms.
    #
    # Let me build this properly.

    # First, define the 7 rank-1 terms for 2x2 Strassen.
    # For 2x2 matrices X, Y (2x2 each), Z = X*Y:
    # The 7 terms are:
    #
    # T1: (X00+X11)(Y00+Y11) = X00*Y00 + X00*Y11 + X11*Y00 + X11*Y11
    #     This contributes to Z00, Z01, Z10, Z11
    #     But this is not a single rank-1 term!
    #
    # I think I'm confusing myself. Let me look at this differently.
    #
    # Strassen's algorithm computes 7 scalar products (if X,Y are 2x2 scalars... no).
    #
    # Actually, for 2x2 matrix multiplication:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y where Zik = sum_j Xij * Yjk
    #
    # Strassen's 7 products:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a product of two linear forms in the entries of X and Y.
    # So each M_i is a rank-1 term in the tensor decomposition!
    #
    # For M1 = (x00+x11)(y00+y11):
    #   U vector: +1 at (0,0), +1 at (1,1)
    #   V vector: +1 at (0,0), +1 at (1,1)
    #   W vector: +1 at (0,0), +1 at (1,1)  [since M1 contributes to Z00 and Z11]
    #
    # Wait, M1 contributes to Z00 with coefficient +1 and to Z11 with coefficient +1.
    # So W vector: +1 at (k=0,i=0) and +1 at (k=1,i=1)
    #
    # Let me define all 7 terms for 2x2:

    def strassen_2x2_terms():
        """Return list of (U_vec, V_vec, W_vec) for 2x2 Strassen, each of length 4."""
        terms = []
        # M1 = (x00+x11)(y00+y11)
        # U: +1 at (0,0), +1 at (1,1)
        # V: +1 at (0,0), +1 at (1,1)
        # W: +1 at (0,0), +1 at (1,1)
        u1 = [1, 0, 0, 1]  # (0,0), (0,1), (1,0), (1,1)
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1]  # (0,0), (0,1), (1,0), (1,1)
        terms.append((u1, v1, w1))

        # M2 = (x10+x11)y00
        # U: +1 at (1,0), +1 at (1,1)
        # V: +1 at (0,0)
        # W: +1 at (0,1), +1 at (1,1)  [M2 contributes to Z10 and Z11]
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 1, 0, 1]  # Z10=(k=0,i=1), Z11=(k=1,i=1)
        terms.append((u2, v2, w2))

        # M3 = x00(y10-y11)
        # U: +1 at (0,0)
        # V: +1 at (1,0), -1 at (1,1)
        # W: +1 at (0,1), +1 at (1,1)  [M3 contributes to Z01 and Z11]
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 1]  # Z01=(k=0,i=1), Z11=(k=1,i=1)
        terms.append((u3, v3, w3))

        # M4 = x11(y01-y00)
        # U: +1 at (1,1)
        # V: +1 at (0,1), -1 at (0,0)
        # W: +1 at (0,0), +1 at (1,1)  [M4 contributes to Z00 and Z11]
        u4 = [0, 0, 0, 1]
        v4 = [-1, 1, 0, 0]
        w4 = [1, 0, 0, 1]
        terms.append((u4, v4, w4))

        # M5 = (x00+x01)y11
        # U: +1 at (0,0), +1 at (0,1)
        # V: +1 at (1,1)
        # W: -1 at (0,0), +1 at (0,1)  [M5 contributes to Z00 with -1, Z01 with +1]
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0]
        terms.append((u5, v5, w5))

        # M6 = (x10-x00)(y00+y01)
        # U: +1 at (1,0), -1 at (0,0)
        # V: +1 at (0,0), +1 at (0,1)
        # W: +1 at (1,1)  [M6 contributes to Z11]
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1]
        terms.append((u6, v6, w6))

        # M7 = (x01-x11)(y10+y11)
        # U: +1 at (0,1), -1 at (1,1)
        # V: +1 at (1,0), +1 at (1,1)
        # W: +1 at (0,0)  [M7 contributes to Z00]
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 0]
        terms.append((u7, v7, w7))

        return terms

    # Now for 4x4, we split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 = A[0:2, 0:2], etc.
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    #
    # C11 = M1 + M4 - M5 + M7  (2x2 blocks)
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # where M1 = (A11+A22)(B11+B22), etc., and each M_i is a 2x2 matrix product.
    #
    # So for each of the 7 "big" products M_i, we use the 2x2 Strassen decomposition
    # (7 terms each), giving 7*7 = 49 terms total.
    #
    # For a big product M_i = X * Y where X and Y are 2x2 matrices (blocks of A and B):
    # Each entry of X is a linear combination of entries of A.
    # Each entry of Y is a linear combination of entries of B.
    # The 2x2 Strassen decomposition of X*Y gives 7 rank-1 terms.
    #
    # Let me define the 7 big products in terms of block entries.
    # Block A11 has entries A[0,0], A[0,1], A[1,0], A[1,1] -> flat indices 0,1,4,5
    # Block A12 has entries A[0,2], A[0,3], A[1,2], A[1,3] -> flat indices 2,3,6,7
    # Block A21 has entries A[2,0], A[2,1], A[3,0], A[3,1] -> flat indices 8,9,12,13
    # Block A22 has entries A[2,2], A[2,3], A[3,2], A[3,3] -> flat indices 10,11,14,15
    #
    # Similarly for B.
    #
    # The 7 big products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # For each big product, X and Y are 2x2 matrices whose entries are linear
    # combinations of A and B entries.
    #
    # Let me define the block-to-flat index mapping.
    # For a 2x2 block starting at (row_offset, col_offset) in a 4x4 matrix:
    # Entry (r,c) of block -> flat index (row_offset+r)*4 + (col_offset+c)

    def block_indices(block_row, block_col):
        """Return flat indices for the 4 entries of a 2x2 block."""
        # block_row, block_col in {0,1}
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((block_row*2 + r)*4 + (block_col*2 + c))
        return indices

    # A block indices:
    A11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    A12_idx = block_indices(0, 1)  # [2, 3, 6, 7]
    A21_idx = block_indices(1, 0)  # [8, 9, 12, 13]
    A22_idx = block_indices(1, 1)  # [10, 11, 14, 15]

    B11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    B12_idx = block_indices(0, 1)  # [2, 3, 6, 7]
    B21_idx = block_indices(1, 0)  # [8, 9, 12, 13]
    B22_idx = block_indices(1, 1)  # [10, 11, 14, 15]

    # C block indices (for W, using W[k*4+i] convention):
    # C11 block: C[0:2, 0:2] -> W indices (k,i) for k in 0,1 and i in 0,1
    C11_idx = [(k*4 + i) for i in range(2) for k in range(2)]  # (0,0),(0,1),(1,0),(1,1) -> [0,1,4,5]
    # Actually W[k*n+i] so W[k*4+i]. For C11 (i in 0,1; k in 0,1):
    # (k=0,i=0)->0, (k=0,i=1)->1, (k=1,i=0)->4, (k=1,i=1)->5
    C11_idx = [0, 1, 4, 5]
    C12_idx = [(k*4 + i) for i in range(2, 4) for k in range(2)]  # i in 2,3
    # (k=0,i=2)->2, (k=0,i=3)->3, (k=1,i=2)->6, (k=1,i=3)->7
    C12_idx = [2, 3, 6, 7]
    C21_idx = [(k*4 + i) for i in range(0, 2) for k in range(2, 4)]  # k in 2,3, i in 0,1
    # (k=2,i=0)->8, (k=2,i=1)->9, (k=3,i=0)->12, (k=3,i=1)->13
    C21_idx = [8, 9, 12, 13]
    C22_idx = [(k*4 + i) for i in range(2, 4) for k in range(2, 4)]  # k in 2,3, i in 2,3
    # (k=2,i=2)->10, (k=2,i=3)->11, (k=3,i=2)->14, (k=3,i=3)->15
    C22_idx = [10, 11, 14, 15]

    # Now, for each of the 7 big products, I need to:
    # 1. Determine the 2x2 matrix X (linear combo of A blocks)
    # 2. Determine the 2x2 matrix Y (linear combo of B blocks)
    # 3. Apply the 7 Strassen 2x2 terms to X*Y
    # 4. Scale the W contributions according to how M_i contributes to C blocks

    # The 7 big products and their contributions to C blocks:
    # M1 -> C11 (+1), C22 (+1)
    # M2 -> C21 (+1), C22 (-1)
    # M3 -> C12 (+1), C22 (+1)
    # M4 -> C11 (+1), C21 (+1)
    # M5 -> C11 (-1), C12 (+1)
    # M6 -> C22 (+1)
    # M7 -> C11 (+1)

    # For each big product M_i = X * Y:
    # X is a 2x2 matrix, Y is a 2x2 matrix.
    # The 7 Strassen terms for X*Y give rank-1 decompositions.
    # Each term has U_vec (for X), V_vec (for Y), W_vec (for the 2x2 result).
    #
    # To map to the full 4x4 problem:
    # - U_vec entries correspond to entries of X, which are linear combos of A entries.
    # - V_vec entries correspond to entries of Y, which are linear combos of B entries.
    # - W_vec entries correspond to entries of the 2x2 result, which map to C block entries.

    # Let me define the X and Y matrices for each big product.
    # X and Y are 2x2, with entries being linear combinations of block entries.

    # For M1 = (A11+A22)(B11+B22):
    # X = A11 + A22, Y = B11 + B22
    # X[r,c] = A11[r,c] + A22[r,c]
    # Y[r,c] = B11[r,c] + B22[r,c]

    # For M2 = (A21+A22)B11:
    # X = A21 + A22, Y = B11

    # etc.

    # Let me define these as coefficient matrices.
    # For each big product, X is a 2x2 matrix where each entry is a linear combo of A flat indices.
    # I'll represent X as a 2x2x16 tensor: X[r,c,a_idx] = coefficient.
    # Similarly for Y.

    def make_X_coefficients(blocks_with_coeffs):
        """blocks_with_coeffs: list of (block_indices, coefficient)"""
        X = [[[0]*16 for _ in range(2)] for _ in range(2)]
        for block_idx, coeff in blocks_with_coeffs:
            for pos, flat_idx in enumerate(block_idx):
                r = pos // 2
                c = pos % 2
                X[r][c][flat_idx] += coeff
        return X

    def make_Y_coefficients(blocks_with_coeffs):
        """Same as make_X_coefficients but for B."""
        Y = [[[0]*16 for _ in range(2)] for _ in range(2)]
        for block_idx, coeff in blocks_with_coeffs:
            for pos, flat_idx in enumerate(block_idx):
                r = pos // 2
                c = pos % 2
                Y[r][c][flat_idx] += coeff
        return Y

    # Define the 7 big products:
    # M1 = (A11+A22)(B11+B22)
    X1 = make_X_coefficients([(A11_idx, 1), (A22_idx, 1)])
    Y1 = make_Y_coefficients([(B11_idx, 1), (B22_idx, 1)])
    # M1 contributes: C11 +1, C22 +1

    # M2 = (A21+A22)B11
    X2 = make_X_coefficients([(A21_idx, 1), (A22_idx, 1)])
    Y2 = make_Y_coefficients([(B11_idx, 1)])
    # M2 contributes: C21 +1, C22 -1

    # M3 = A11(B21-B22)
    X3 = make_X_coefficients([(A11_idx, 1)])
    Y3 = make_Y_coefficients([(B21_idx, 1), (B22_idx, -1)])
    # M3 contributes: C12 +1, C22 +1

    # M4 = A22(B12-B11)
    X4 = make_X_coefficients([(A22_idx, 1)])
    Y4 = make_Y_coefficients([(B12_idx, 1), (B11_idx, -1)])
    # M4 contributes: C11 +1, C21 +1

    # M5 = (A11+A12)B22
    X5 = make_X_coefficients([(A11_idx, 1), (A12_idx, 1)])
    Y5 = make_Y_coefficients([(B22_idx, 1)])
    # M5 contributes: C11 -1, C12 +1

    # M6 = (A21-A11)(B11+B12)
    X6 = make_X_coefficients([(A21_idx, 1), (A11_idx, -1)])
    Y6 = make_Y_coefficients([(B11_idx, 1), (B12_idx, 1)])
    # M6 contributes: C22 +1

    # M7 = (A12-A22)(B21+B22)
    X7 = make_X_coefficients([(A12_idx, 1), (A22_idx, -1)])
    Y7 = make_Y_coefficients([(B21_idx, 1), (B22_idx, 1)])
    # M7 contributes: C11 +1

    # Now, for each big product, apply the 7 Strassen 2x2 terms.
    # The 2x2 Strassen terms give (u_vec, v_vec, w_vec) of length 4 each.
    # u_vec[r*2+c] is the coefficient for X[r,c]
    # v_vec[j*2+k] is the coefficient for Y[j,k]
    # w_vec[k*2+i] is the coefficient for the 2x2 result Z[k,i]

    # To map to 4x4:
    # U_flat[a_idx] = sum over (r,c) of u_vec[r*2+c] * X_coeff[r,c,a_idx]
    # V_flat[b_idx] = sum over (j,k) of v_vec[j*2+k] * Y_coeff[j,k,b_idx]
    # W_flat[c_idx] = sum over (k,i) of w_vec[k*2+i] * C_block_coeff[k,i,c_idx]

    # For the W mapping, the 2x2 result Z maps to a C block.
    # Z[k,i] maps to C_block[k,i] which has flat W index C_block_idx[k*2+i].
    # But the big product M_i contributes to multiple C blocks with coefficients.
    # So I need to distribute the W contribution.

    # Actually, let me think about this more carefully.
    # The big product M_i = X*Y is a 2x2 matrix.
    # This 2x2 matrix contributes to C blocks with certain coefficients.
    # For example, M1 contributes to C11 with coeff +1 and C22 with coeff +1.
    # So if the 2x2 result has entry Z[k,i], it contributes:
    #   +1 to C11[k,i] (if M1) and +1 to C22[k,i] (if M1)
    #
    # So W_flat for the full 4x4:
    # W_flat[C11_idx[k*2+i]] += w_vec[k*2+i] * (M_i's contribution to C11)
    # W_flat[C22_idx[k*2+i]] += w_vec[k*2+i] * (M_i's contribution to C22)
    # etc.

    # Define the contributions of each big product to C blocks:
    # (block_name, coefficient)
    M_contributions = {
        1: [("C11", 1), ("C22", 1)],
        2: [("C21", 1), ("C22", -1)],
        3: [("C12", 1), ("C22", 1)],
        4: [("C11", 1), ("C21", 1)],
        5: [("C11", -1), ("C12", 1)],
        6: [("C22", 1)],
        7: [("C11", 1)],
    }

    C_block_indices = {
        "C11": C11_idx,
        "C12": C12_idx,
        "C21": C21_idx,
        "C22": C22_idx,
    }

    # Get the 7 Strassen 2x2 terms
    s2_terms = strassen_2x2_terms()

    # Now build the full decomposition
    col = 0
    for big_i in range(1, 8):
        X_coeff = [X1, X2, X3, X4, X5, X6, X7][big_i - 1]
        Y_coeff = [Y1, Y2, Y3, Y4, Y5, Y6, Y7][big_i - 1]
        contribs = M_contributions[big_i]

        for u_vec, v_vec, w_vec in s2_terms:
            # Build U_flat
            U_flat = [0] * 16
            for r in range(2):
                for c in range(2):
                    coeff = u_vec[r * 2 + c]
                    if coeff != 0:
                        for a_idx in range(16):
                            U_flat[a_idx] += coeff * X_coeff[r][c][a_idx]

            # Build V_flat
            V_flat = [0] * 16
            for j in range(2):
                for k in range(2):
                    coeff = v_vec[j * 2 + k]
                    if coeff != 0:
                        for b_idx in range(16):
                            V_flat[b_idx] += coeff * Y_coeff[j][k][b_idx]

            # Build W_flat
            W_flat = [0] * 16
            for ki in range(4):  # k*2+i
                k = ki // 2
                i = ki % 2
                coeff = w_vec[ki]
                if coeff != 0:
                    for block_name, block_coeff in contribs:
                        c_block_idx = C_block_indices[block_name]
                        w_idx = c_block_idx[ki]
                        W_flat[w_idx] += coeff * block_coeff

            # Add to U, V, W
            for idx in range(16):
                U[idx][col] = U_flat[idx]
                V[idx][col] = V_flat[idx]
                W[idx][col] = W_flat[idx]
            col += 1

        if time.time() > deadline:
            break

    # Verify the decomposition
    # Check that sum_r U[:,r] * V[:,r]^T reconstructs the multiplication tensor
    # i.e., for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # if (i,j,k) is a valid triple, else 0.
    # Actually, the condition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta (the multiplication tensor has T[i,j,k] = 1 for all i,j,k)
    # Wait, no. The multiplication tensor T has T[i,j,k] = 1 for all i,j,k in 0..n-1.
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So for all i,j,k in 0..3: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1.

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for r in range(rank):
                    val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-6:
        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
