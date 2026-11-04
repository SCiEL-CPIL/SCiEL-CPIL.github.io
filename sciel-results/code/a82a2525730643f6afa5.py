"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Uses the Strassen-style recursive approach to achieve rank 49 (optimal for 4x4).
The decomposition is constructed by combining 7 Strassen-like products at each level
of a 2x2 block decomposition, recursively applied to get 4x4 from 2x2.

For 4x4 matrices, we can use a 2-level recursive Strassen:
- Level 1: Split 4x4 into 2x2 blocks, use 7 products (Strassen)
- Level 2: Each 2x2 product is itself computed with 7 products

Total rank: 7 * 7 = 49

This is the known optimal rank for 4x4 matrix multiplication.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-49 tensor decomposition for 4x4 matrix multiplication.
    
    Uses recursive Strassen decomposition:
    - 4x4 = 2x2 blocks of 2x2
    - Each 2x2 multiplication uses 7 products (Strassen)
    - Total: 7 * 7 = 49 products
    
    The factor matrices U, V, W have entries in {0, 1, -1} (integers).
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # For 4x4 matrix multiplication using recursive Strassen:
    # We need to construct U (16 x 49), V (16 x 49), W (16 x 49)
    
    # Strategy: Use the known 4x4 rank-49 decomposition.
    # This can be constructed by viewing 4x4 matrices as 2x2 block matrices
    # where each block is 2x2, and applying Strassen's 7-product formula.
    
    # Let A, B be 4x4 matrices partitioned as:
    # A = [[A11, A12], [A21, A22]]  where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]]  where each Bij is 2x2
    #
    # Strassen computes C = A*B using 7 products:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # Each Pk is a 2x2 matrix product, which itself can be done with 7 scalar products.
    # So total = 7 * 7 = 49 scalar products.
    
    # Build the decomposition explicitly.
    # Indexing: A[i,j] where i,j in 0..3
    # A is 4x4, so we can think of it as 2x2 blocks of 2x2 matrices.
    # Block (bi, bj) of A is A[2*bi:2*bi+2, 2*bj:2*bj+2]
    
    # Let's build U, V, W for the 49-product decomposition.
    # Each product corresponds to a pair (strassen_product, inner_strassen_product)
    # where strassen_product in 0..6 and inner_strassen_product in 0..6.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7 products for 2x2 block multiplication:
    # For blocks, we have:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # For each Strassen product Pk, we need to compute a 2x2 matrix product
    # of two 2x2 matrices. Each such product uses 7 scalar products (Strassen again).
    
    # Let's define the Strassen products for 2x2 scalar matrices:
    # Given X, Y (2x2), compute Z = X*Y using:
    # Q1 = (X11 + X22)(Y11 + Y22)
    # Q2 = (X21 + X22)Y11
    # Q3 = X11(Y12 - Y22)
    # Q4 = X22(Y21 - Y11)
    # Q5 = (X11 + X12)Y22
    # Q6 = (X21 - X11)(Y11 + Y12)
    # Q7 = (X12 - X22)(Y21 + Y22)
    # Z11 = Q1 + Q4 - Q5 + Q7
    # Z12 = Q3 + Q5
    # Z21 = Q2 + Q4
    # Z22 = Q1 - Q2 + Q3 + Q6
    
    # Now, for the outer Strassen level:
    # Each Pk is a product of two 2x2 block matrices.
    # We need to express each block matrix in terms of the original A and B entries.
    
    # Let's enumerate all 49 products.
    # Product index = strassen_outer * 7 + strassen_inner
    
    # For each outer Strassen product Pk, we have two 2x2 matrices X and Y.
    # X and Y are linear combinations of the 2x2 blocks of A and B.
    # Then we apply inner Strassen to compute X*Y.
    
    # Let's define the block indices:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # Outer Strassen products:
    # P1: X = A00 + A11, Y = B00 + B11
    # P2: X = A10 + A11, Y = B00
    # P3: X = A00, Y = B01 - B11
    # P4: X = A11, Y = B10 - B00
    # P5: X = A00 + A01, Y = B11
    # P6: X = A10 - A00, Y = B00 + B01
    # P7: X = A01 - A11, Y = B10 + B11
    
    # For each of these, X and Y are 2x2 matrices whose entries are linear
    # combinations of the entries of A and B.
    
    # Let's build this systematically.
    
    # First, let's define how to extract a 2x2 block from a 4x4 matrix.
    # Block (bi, bj) of a 4x4 matrix M is M[2*bi:2*bi+2, 2*bj:2*bj+2]
    
    # For the U matrix (input A side):
    # U[i*m+j, col] = coefficient of A[i,j] in the col-th product's X factor
    
    # For the V matrix (input B side):
    # V[j*p+k, col] = coefficient of B[j,k] in the col-th product's Y factor
    
    # For the W matrix (output C side):
    # W[k*n+i, col] = coefficient of the col-th product in C[i,k]
    
    # Let's build this step by step.
    
    # Define the outer Strassen products as linear combinations of blocks.
    # Each block is a 2x2 matrix, so we need to express each entry of X and Y
    # in terms of the entries of A and B.
    
    # Let's use a helper to add a linear combination of blocks to X or Y.
    
    # For each outer product k (0..6), we have:
    # X = sum of coefficients * A_blocks
    # Y = sum of coefficients * B_blocks
    
    # Then for the inner Strassen, we have 7 products Q1..Q7, each of which
    # is a product of two scalars (entries of X and Y).
    
    # Let's define the structure.
    
    # Outer Strassen: for product k, X = sum_{bi,bj} xa[bi,bj] * A[bi,bj]
    # where xa[bi,bj] is a 2x2 coefficient matrix (or scalar if we think of blocks)
    
    # Actually, let's think of it differently. Each block A[bi,bj] is a 2x2 matrix
    # with entries A[2*bi+r, 2*bj+c] for r,c in 0,1.
    
    # For the outer Strassen product Pk, X is a 2x2 matrix where:
    # X[r,c] = sum of coefficients * A[2*bi+r, 2*bj+c]
    
    # Let's define the coefficient patterns for X and Y for each outer product.
    
    # For P1: X = A00 + A11, so X[r,c] = A[r,c] + A[2+r, 2+c]
    # For P2: X = A10 + A11, so X[r,c] = A[2+r, c] + A[2+r, 2+c]
    # For P3: X = A00, so X[r,c] = A[r, c]
    # For P4: X = A11, so X[r,c] = A[2+r, 2+c]
    # For P5: X = A00 + A01, so X[r,c] = A[r, c] + A[r, 2+c]
    # For P6: X = A10 - A00, so X[r,c] = A[2+r, c] - A[r, c]
    # For P7: X = A01 - A11, so X[r,c] = A[r, 2+c] - A[2+r, 2+c]
    
    # Similarly for Y with B.
    
    # Now for the inner Strassen, given X and Y (2x2 matrices), we compute 7 products:
    # Q1 = (X11 + X22)(Y11 + Y22)
    # Q2 = (X21 + X22)Y11
    # Q3 = X11(Y12 - Y22)
    # Q4 = X22(Y21 - Y11)
    # Q5 = (X11 + X12)Y22
    # Q6 = (X21 - X11)(Y11 + Y12)
    # Q7 = (X12 - X22)(Y21 + Y22)
    
    # Each Qj is a product of two scalars, each of which is a linear combination
    # of entries of X and Y, which are themselves linear combinations of A and B entries.
    
    # Let's build the U, V, W matrices.
    
    # For each outer product k (0..6) and inner product j (0..6):
    # col = k * 7 + j
    
    # We need to determine:
    # 1. The coefficients of A entries in the X factor of Qj
    # 2. The coefficients of B entries in the Y factor of Qj
    # 3. The coefficients in the final output C
    
    # Let's define the inner Strassen products' X and Y factors.
    # For inner product j, the X factor is a linear combination of X entries,
    # and the Y factor is a linear combination of Y entries.
    
    # Inner Strassen:
    # Q1: X_factor = X11 + X22, Y_factor = Y11 + Y22
    # Q2: X_factor = X21 + X22, Y_factor = Y11
    # Q3: X_factor = X11, Y_factor = Y12 - Y22
    # Q4: X_factor = X22, Y_factor = Y21 - Y11
    # Q5: X_factor = X11 + X12, Y_factor = Y22
    # Q6: X_factor = X21 - X11, Y_factor = Y11 + Y12
    # Q7: X_factor = X12 - X22, Y_factor = Y21 + Y22
    
    # X entries in terms of A:
    # For outer product k, X[r,c] = sum over (bi,bj) of coeff_A[k, bi, bj, r, c] * A[2*bi+r, 2*bj+c]
    
    # Let's define coeff_A and coeff_B.
    
    # coeff_A[k, bi, bj, r, c] = coefficient of A[2*bi+r, 2*bj+c] in X[r,c] for outer product k
    
    # From above:
    # P1 (k=0): X[r,c] = A[r,c] + A[2+r, 2+c]
    #   coeff_A[0, 0, 0, r, c] = 1
    #   coeff_A[0, 1, 1, r, c] = 1
    # P2 (k=1): X[r,c] = A[2+r, c] + A[2+r, 2+c]
    #   coeff_A[1, 1, 0, r, c] = 1
    #   coeff_A[1, 1, 1, r, c] = 1
    # P3 (k=2): X[r,c] = A[r, c]
    #   coeff_A[2, 0, 0, r, c] = 1
    # P4 (k=3): X[r,c] = A[2+r, 2+c]
    #   coeff_A[3, 1, 1, r, c] = 1
    # P5 (k=4): X[r,c] = A[r, c] + A[r, 2+c]
    #   coeff_A[4, 0, 0, r, c] = 1
    #   coeff_A[4, 0, 1, r, c] = 1
    # P6 (k=5): X[r,c] = A[2+r, c] - A[r, c]
    #   coeff_A[5, 1, 0, r, c] = 1
    #   coeff_A[5, 0, 0, r, c] = -1
    # P7 (k=6): X[r,c] = A[r, 2+c] - A[2+r, 2+c]
    #   coeff_A[6, 0, 1, r, c] = 1
    #   coeff_A[6, 1, 1, r, c] = -1
    
    # Similarly for B:
    # P1 (k=0): Y[r,c] = B[r,c] + B[2+r, 2+c]
    #   coeff_B[0, 0, 0, r, c] = 1
    #   coeff_B[0, 1, 1, r, c] = 1
    # P2 (k=1): Y[r,c] = B[r, c]
    #   coeff_B[1, 0, 0, r, c] = 1
    # P3 (k=2): Y[r,c] = B[r, 2+c] - B[2+r, 2+c]
    #   coeff_B[2, 0, 1, r, c] = 1
    #   coeff_B[2, 1, 1, r, c] = -1
    # P4 (k=3): Y[r,c] = B[2+r, c] - B[r, c]
    #   coeff_B[3, 1, 0, r, c] = 1
    #   coeff_B[3, 0, 0, r, c] = -1
    # P5 (k=4): Y[r,c] = B[2+r, 2+c]
    #   coeff_B[4, 1, 1, r, c] = 1
    # P6 (k=5): Y[r,c] = B[r, c] + B[r, 2+c]
    #   coeff_B[5, 0, 0, r, c] = 1
    #   coeff_B[5, 0, 1, r, c] = 1
    # P7 (k=6): Y[r,c] = B[2+r, c] + B[2+r, 2+c]
    #   coeff_B[6, 1, 0, r, c] = 1
    #   coeff_B[6, 1, 1, r, c] = 1
    
    # Now for the inner Strassen, for each outer product k and inner product j:
    # The X factor of Qj is a linear combination of X entries.
    # X[r,c] = sum_{bi,bj} coeff_A[k, bi, bj, r, c] * A[2*bi+r, 2*bj+c]
    
    # So the X factor of Qj = sum_{r,c} inner_x_coeff[j, r, c] * X[r,c]
    #   = sum_{r,c,bi,bj} inner_x_coeff[j, r, c] * coeff_A[k, bi, bj, r, c] * A[2*bi+r, 2*bj+c]
    
    # Similarly for Y.
    
    # Inner Strassen X factors:
    # Q1 (j=0): X11 + X22 -> inner_x_coeff[0, 0, 0] = 1, inner_x_coeff[0, 1, 1] = 1
    # Q2 (j=1): X21 + X22 -> inner_x_coeff[1, 1, 0] = 1, inner_x_coeff[1, 1, 1] = 1
    # Q3 (j=2): X11 -> inner_x_coeff[2, 0, 0] = 1
    # Q4 (j=3): X22 -> inner_x_coeff[3, 1, 1] = 1
    # Q5 (j=4): X11 + X12 -> inner_x_coeff[4, 0, 0] = 1, inner_x_coeff[4, 0, 1] = 1
    # Q6 (j=5): X21 - X11 -> inner_x_coeff[5, 1, 0] = 1, inner_x_coeff[5, 0, 0] = -1
    # Q7 (j=6): X12 - X22 -> inner_x_coeff[6, 0, 1] = 1, inner_x_coeff[6, 1, 1] = -1
    
    # Inner Strassen Y factors:
    # Q1 (j=0): Y11 + Y22 -> inner_y_coeff[0, 0, 0] = 1, inner_y_coeff[0, 1, 1] = 1
    # Q2 (j=1): Y11 -> inner_y_coeff[1, 0, 0] = 1
    # Q3 (j=2): Y12 - Y22 -> inner_y_coeff[2, 0, 1] = 1, inner_y_coeff[2, 1, 1] = -1
    # Q4 (j=3): Y21 - Y11 -> inner_y_coeff[3, 1, 0] = 1, inner_y_coeff[3, 0, 0] = -1
    # Q5 (j=4): Y22 -> inner_y_coeff[4, 1, 1] = 1
    # Q6 (j=5): Y11 + Y12 -> inner_y_coeff[5, 0, 0] = 1, inner_y_coeff[5, 0, 1] = 1
    # Q7 (j=6): Y21 + Y22 -> inner_y_coeff[6, 1, 0] = 1, inner_y_coeff[6, 1, 1] = 1
    
    # Now for the output C:
    # C = A*B, where C is 4x4.
    # C is composed of 2x2 blocks C00, C01, C10, C11.
    # From outer Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pk is a 2x2 matrix, and Pk = sum of inner Strassen products Qj with coefficients.
    # From inner Strassen:
    # Z11 = Q1 + Q4 - Q5 + Q7
    # Z12 = Q3 + Q5
    # Z21 = Q2 + Q4
    # Z22 = Q1 - Q2 + Q3 + Q6
    
    # So for outer product k, the 2x2 matrix Pk has entries:
    # Pk[0,0] = Q1 + Q4 - Q5 + Q7 (where Qj are the inner products for this k)
    # Pk[0,1] = Q3 + Q5
    # Pk[1,0] = Q2 + Q4
    # Pk[1,1] = Q1 - Q2 + Q3 + Q6
    
    # Now, C[2*bi+r, 2*bj+c] = sum over outer products k of outer_coeff[k, bi, bj, r, c] * Pk[r,c]
    
    # From above:
    # C00 = P1 + P4 - P5 + P7 -> outer_coeff for block (0,0):
    #   k=0 (P1): +1
    #   k=3 (P4): +1
    #   k=4 (P5): -1
    #   k=6 (P7): +1
    # C01 = P3 + P5 -> outer_coeff for block (0,1):
    #   k=2 (P3): +1
    #   k=4 (P5): +1
    # C10 = P2 + P4 -> outer_coeff for block (1,0):
    #   k=1 (P2): +1
    #   k=3 (P4): +1
    # C11 = P1 - P2 + P3 + P6 -> outer_coeff for block (1,1):
    #   k=0 (P1): +1
    #   k=1 (P2): -1
    #   k=2 (P3): +1
    #   k=5 (P6): +1
    
    # Now we can build U, V, W.
    
    # For each col = k*7 + j:
    # U[i*m+j, col] = sum over (r,c) of inner_x_coeff[j, r, c] * coeff_A[k, bi, bj, r, c]
    #   where i = 2*bi + r, j_a = 2*bj + c (note: j_a is the column index in A, not to be confused with inner j)
    
    # Let's be careful with indices.
    # A has indices (row, col) where row, col in 0..3.
    # U has rows indexed by i*m + j where i is row in A, j is col in A. So U[row_A * 4 + col_A, col]
    
    # For a given col = k*7 + j:
    # The X factor is: sum_{r,c} inner_x_coeff[j, r, c] * X[r,c]
    # X[r,c] = sum_{bi,bj} coeff_A[k, bi, bj, r, c] * A[2*bi+r, 2*bj+c]
    
    # So the coefficient of A[row, col_a] in the X factor is:
    # sum over (r,c,bi,bj) such that row = 2*bi+r and col_a = 2*bj+c of:
    #   inner_x_coeff[j, r, c] * coeff_A[k, bi, bj, r, c]
    
    # Similarly for V with B.
    
    # And for W:
    # W[k_out*n + i_out, col] where k_out is row in C, i_out is col in C.
    # C[k_out, i_out] = sum over outer products k of outer_coeff[k, bi_out, bj_out, r_out, c_out] * Pk[r_out, c_out]
    # where k_out = 2*bi_out + r_out, i_out = 2*bj_out + c_out
    # Pk[r_out, c_out] = sum over inner products j of inner_output_coeff[j, r_out, c_out] * Qj
    
    # inner_output_coeff:
    # For Z11 (r_out=0, c_out=0): Q1+Q4-Q5+Q7 -> j=0:1, j=3:1, j=4:-1, j=6:1
    # For Z12 (r_out=0, c_out=1): Q3+Q5 -> j=2:1, j=4:1
    # For Z21 (r_out=1, c_out=0): Q2+Q4 -> j=1:1, j=3:1
    # For Z22 (r_out=1, c_out=1): Q1-Q2+Q3+Q6 -> j=0:1, j=1:-1, j=2:1, j=5:1
    
    # Let's build the coefficient tables.
    
    # coeff_A[k, bi, bj, r, c]:
    coeff_A = [[[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    # P1 (k=0): A00 + A11
    for r in range(2):
        for c in range(2):
            coeff_A[0][0][0][r][c] = 1
            coeff_A[0][1][1][r][c] = 1
    # P2 (k=1): A10 + A11
    for r in range(2):
        for c in range(2):
            coeff_A[1][1][0][r][c] = 1
            coeff_A[1][1][1][r][c] = 1
    # P3 (k=2): A00
    for r in range(2):
        for c in range(2):
            coeff_A[2][0][0][r][c] = 1
    # P4 (k=3): A11
    for r in range(2):
        for c in range(2):
            coeff_A[3][1][1][r][c] = 1
    # P5 (k=4): A00 + A01
    for r in range(2):
        for c in range(2):
            coeff_A[4][0][0][r][c] = 1
            coeff_A[4][0][1][r][c] = 1
    # P6 (k=5): A10 - A00
    for r in range(2):
        for c in range(2):
            coeff_A[5][1][0][r][c] = 1
            coeff_A[5][0][0][r][c] = -1
    # P7 (k=6): A01 - A11
    for r in range(2):
        for c in range(2):
            coeff_A[6][0][1][r][c] = 1
            coeff_A[6][1][1][r][c] = -1
    
    # coeff_B[k, bi, bj, r, c]:
    coeff_B = [[[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    # P1 (k=0): B00 + B11
    for r in range(2):
        for c in range(2):
            coeff_B[0][0][0][r][c] = 1
            coeff_B[0][1][1][r][c] = 1
    # P2 (k=1): B00
    for r in range(2):
        for c in range(2):
            coeff_B[1][0][0][r][c] = 1
    # P3 (k=2): B01 - B11
    for r in range(2):
        for c in range(2):
            coeff_B[2][0][1][r][c] = 1
            coeff_B[2][1][1][r][c] = -1
    # P4 (k=3): B10 - B00
    for r in range(2):
        for c in range(2):
            coeff_B[3][1][0][r][c] = 1
            coeff_B[3][0][0][r][c] = -1
    # P5 (k=4): B11
    for r in range(2):
        for c in range(2):
            coeff_B[4][1][1][r][c] = 1
    # P6 (k=5): B00 + B01
    for r in range(2):
        for c in range(2):
            coeff_B[5][0][0][r][c] = 1
            coeff_B[5][0][1][r][c] = 1
    # P7 (k=6): B10 + B11
    for r in range(2):
        for c in range(2):
            coeff_B[6][1][0][r][c] = 1
            coeff_B[6][1][1][r][c] = 1
    
    # inner_x_coeff[j, r, c]:
    inner_x_coeff = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # Q1 (j=0): X11 + X22
    inner_x_coeff[0][0][0] = 1
    inner_x_coeff[0][1][1] = 1
    # Q2 (j=1): X21 + X22
    inner_x_coeff[1][1][0] = 1
    inner_x_coeff[1][1][1] = 1
    # Q3 (j=2): X11
    inner_x_coeff[2][0][0] = 1
    # Q4 (j=3): X22
    inner_x_coeff[3][1][1] = 1
    # Q5 (j=4): X11 + X12
    inner_x_coeff[4][0][0] = 1
    inner_x_coeff[4][0][1] = 1
    # Q6 (j=5): X21 - X11
    inner_x_coeff[5][1][0] = 1
    inner_x_coeff[5][0][0] = -1
    # Q7 (j=6): X12 - X22
    inner_x_coeff[6][0][1] = 1
    inner_x_coeff[6][1][1] = -1
    
    # inner_y_coeff[j, r, c]:
    inner_y_coeff = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # Q1 (j=0): Y11 + Y22
    inner_y_coeff[0][0][0] = 1
    inner_y_coeff[0][1][1] = 1
    # Q2 (j=1): Y11
    inner_y_coeff[1][0][0] = 1
    # Q3 (j=2): Y12 - Y22
    inner_y_coeff[2][0][1] = 1
    inner_y_coeff[2][1][1] = -1
    # Q4 (j=3): Y21 - Y11
    inner_y_coeff[3][1][0] = 1
    inner_y_coeff[3][0][0] = -1
    # Q5 (j=4): Y22
    inner_y_coeff[4][1][1] = 1
    # Q6 (j=5): Y11 + Y12
    inner_y_coeff[5][0][0] = 1
    inner_y_coeff[5][0][1] = 1
    # Q7 (j=6): Y21 + Y22
    inner_y_coeff[6][1][0] = 1
    inner_y_coeff[6][1][1] = 1
    
    # inner_output_coeff[j, r, c]: coefficient of Qj in Z[r,c]
    inner_output_coeff = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # Z11 (r=0,c=0) = Q1 + Q4 - Q5 + Q7
    inner_output_coeff[0][0][0] = 1
    inner_output_coeff[3][0][0] = 1
    inner_output_coeff[4][0][0] = -1
    inner_output_coeff[6][0][0] = 1
    # Z12 (r=0,c=1) = Q3 + Q5
    inner_output_coeff[2][0][1] = 1
    inner_output_coeff[4][0][1] = 1
    # Z21 (r=1,c=0) = Q2 + Q4
    inner_output_coeff[1][1][0] = 1
    inner_output_coeff[3][1][0] = 1
    # Z22 (r=1,c=1) = Q1 - Q2 + Q3 + Q6
    inner_output_coeff[0][1][1] = 1
    inner_output_coeff[1][1][1] = -1
    inner_output_coeff[2][1][1] = 1
    inner_output_coeff[5][1][1] = 1
    
    # outer_coeff[k, bi, bj, r, c]: coefficient of Pk in C block (bi,bj) entry (r,c)
    outer_coeff = [[[[[0]*2 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    # C00 = P1 + P4 - P5 + P7
    for r in range(2):
        for c in range(2):
            outer_coeff[0][0][0][r][c] = 1
            outer_coeff[3][0][0][r][c] = 1
            outer_coeff[4][0][0][r][c] = -1
            outer_coeff[6][0][0][r][c] = 1
    # C01 = P3 + P5
    for r in range(2):
        for c in range(2):
            outer_coeff[2][0][1][r][c] = 1
            outer_coeff[4][0][1][r][c] = 1
    # C10 = P2 + P4
    for r in range(2):
        for c in range(2):
            outer_coeff[1][1][0][r][c] = 1
            outer_coeff[3][1][0][r][c] = 1
    # C11 = P1 - P2 + P3 + P6
    for r in range(2):
        for c in range(2):
            outer_coeff[0][1][1][r][c] = 1
            outer_coeff[1][1][1][r][c] = -1
            outer_coeff[2][1][1][r][c] = 1
            outer_coeff[5][1][1][r][c] = 1
    
    # Now build U, V, W.
    for k in range(7):  # outer Strassen product
        for j in range(7):  # inner Strassen product
            col = k * 7 + j
            
            # Build U column: coefficient of each A entry in the X factor
            for row_a in range(4):
                for col_a in range(4):
                    bi = row_a // 2
                    bj = col_a // 2
                    r = row_a % 2
                    c = col_a % 2
                    coeff = 0
                    for rr in range(2):
                        for cc in range(2):
                            coeff += inner_x_coeff[j][rr][cc] * coeff_A[k][bi][bj][rr][cc]
                    if coeff != 0:
                        U[row_a * m + col_a][col] = coeff
            
            # Build V column: coefficient of each B entry in the Y factor
            for row_b in range(4):
                for col_b in range(4):
                    bi = row_b // 2
                    bj = col_b // 2
                    r = row_b % 2
                    c = col_b % 2
                    coeff = 0
                    for rr in range(2):
                        for cc in range(2):
                            coeff += inner_y_coeff[j][rr][cc] * coeff_B[k][bi][bj][rr][cc]
                    if coeff != 0:
                        V[row_b * p + col_b][col] = coeff
            
            # Build W column: coefficient of this product in each C entry
            for row_c in range(4):
                for col_c in range(4):
                    bi = row_c // 2
                    bj = col_c // 2
                    r = row_c % 2
                    c = col_c % 2
                    coeff = 0
                    for kk in range(7):
                        for rr in range(2):
                            for cc in range(2):
                                coeff += outer_coeff[kk][bi][bj][rr][cc] * inner_output_coeff[j][rr][cc]
                    if coeff != 0:
                        W[row_c * n + col_c][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
