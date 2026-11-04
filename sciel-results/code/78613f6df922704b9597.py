import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Rank 32 solution:
    # 1. 4x4x4 core using Strassen (rank 28)
    # 2. 4x4x1 (5th column) using rank 4

    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # (16, 32)
    V = [[0] * rank for _ in range(m * p)]  # (20, 32)
    W = [[0] * rank for _ in range(p * n)]  # (20, 32)

    col = 0

    def add_term(i, j, k, cu, cv, cw):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] += cu
        V[j * p + k][col] += cv
        W[k * n + i][col] += cw
        col += 1

    # Strassen 7 terms for 4x4x4
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)

    # Each Strassen term is a 2x2 matrix product, decomposed into 4 schoolbook terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For each Strassen term, we need to expand the 2x2 product into 4 rank-1 tensors.
    # The linear forms involve sums/differences of A and B entries.

    # M1 = (A00+A11)(B00+B11)
    # (A00+A11) is a 2x2 matrix with entries:
    #   [A00_00 + A11_00, A00_01 + A11_01]
    #   [A10_00 + A11_10, A10_01 + A11_11]
    # Wait, A11 is rows 2-3, cols 2-3. So A11_00 = A[2,2], A11_01 = A[2,3], etc.
    # (B00+B11) is a 2x2 matrix with entries:
    #   [B00_00 + B11_00, B00_01 + B11_01]
    #   [B10_00 + B11_10, B10_01 + B11_11]

    # The product (A00+A11)(B00+B11) contributes to C00 and C11.
    # Let's define helper to add a 2x2 product term.

    def add_2x2_product(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols,
                        a_coeff, b_coeff, c_coeff):
        """
        Add terms for (sum a_coeff*A[a_rows+i, a_cols+j]) * (sum b_coeff*B[b_rows+j, b_cols+k])
        contributing to c_coeff*C[c_rows+i, c_cols+k]
        """
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A entry: a_rows[i], a_cols[j] with coeff a_coeff
                    # B entry: b_rows[j], b_cols[k] with coeff b_coeff
                    # C entry: c_rows[i], c_cols[k] with coeff c_coeff
                    ai = a_rows[i]
                    aj = a_cols[j]
                    bj = b_rows[j]
                    bk = b_cols[k]
                    ci = c_rows[i]
                    ck = c_cols[k]
                    if ai < n and aj < m and bj < m and bk < p and ci < n and ck < p:
                        add_term(ai, aj, bk, a_coeff, b_coeff, c_coeff)

    # M1 = (A00+A11)(B00+B11)
    # A00+A11: rows [0,2], cols [0,2] (but it's a 2x2 matrix, so we need to be careful)
    # Actually, A00 is at (0,0), A11 is at (2,2). The sum is not a contiguous block.
    # We need to handle this differently.

    # Let me just enumerate all 28 terms explicitly.

    # M1 = (A00+A11)(B00+B11)
    # A00+A11 as a 2x2 matrix:
    #   [A[0,0]+A[2,2], A[0,1]+A[2,3]]
    #   [A[1,0]+A[3,2], A[1,1]+A[3,3]]
    # B00+B11 as a 2x2 matrix:
    #   [B[0,0]+B[2,2], B[0,1]+B[2,3]]
    #   [B[1,0]+B[3,2], B[1,1]+B[3,3]]

    # Product P = (A00+A11)(B00+B11)
    # P[0,0] = (A[0,0]+A[2,2])(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])(B[1,0]+B[3,2])
    # P[0,1] = (A[0,0]+A[2,2])(B[0,1]+B[2,3]) + (A[0,1]+A[2,3])(B[1,1]+B[3,3])
    # P[1,0] = (A[1,0]+A[3,2])(B[0,0]+B[2,2]) + (A[1,1]+A[3,3])(B[1,0]+B[3,2])
    # P[1,1] = (A[1,0]+A[3,2])(B[0,1]+B[2,3]) + (A[1,1]+A[3,3])(B[1,1]+B[3,3])

    # This is getting complex. Let me use a different approach.
    # I'll define the 7 Strassen terms as linear forms and expand each into 4 rank-1 terms.

    # Actually, the standard way is:
    # Each Strassen "multiplication" is of two 2x2 matrices.
    # The 2x2 matrix product has 8 scalar multiplications (4 for each of 4 entries? No, 2x2 matmul has 8 mults).
    # Wait, 2x2 matmul: Z[i,k] = sum_j X[i,j]*Y[j,k]. That's 2*2*2 = 8 scalar multiplications.
    # So 7 Strassen terms * 8 = 56? No, that's not right either.

    # Let me reconsider. The rank of 2x2x2 matmul is 7 (Strassen).
    # So for 4x4x4, if we use 4 independent 2x2x2, that's rank 28.
    # But 4x4x4 is NOT 4 independent 2x2x2.

    # The correct approach: 4x4x4 can be done with Strassen by treating it as a 2x2 block matrix
    # where each block is 2x2. The Strassen algorithm uses 7 block multiplications, each of which
    # is a 2x2x2 matmul (rank 7). Total rank = 7*7 = 49? No, that's not right.

    # Actually, Strassen for 4x4x4: you partition into 2x2 blocks of 2x2 matrices.
    # You do 7 block multiplications, each block multiplication is a 2x2x2 matmul.
    # Each 2x2x2 matmul has rank 7. So total rank = 7 * 7 = 49.

    # Hmm, but I thought 4x4x4 had rank 35 (Lander-Bailey) or 49 (Strassen).
    # Let me just use rank 49 for 4x4x4 and rank 4 for the 5th column. Total rank 53.

    # Actually, let me just implement the schoolbook for 4x4x4 (rank 64) + rank 4 = rank 68.
    # That's simpler and I'm confident it's correct.

    # Wait, the current program uses rank 80 (full schoolbook for 4x4x5).
    # I can at least get rank 68 by doing schoolbook for 4x4x4 (64) + rank 4 for 5th column.

    # Let me do that.

    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Schoolbook for 4x4x4 (first 4 columns of B and C)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Rank 4 for the 5th column
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each j, one term:
    #   U has 1s at (0,j), (1,j), (2,j), (3,j)
    #   V has 1 at (j,4)
    #   W has 1s at (0,4), (1,4), (2,4), (3,4)
    for j in range(m):
        for i in range(n):
            U[i * m + j][col] += 1
        V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
