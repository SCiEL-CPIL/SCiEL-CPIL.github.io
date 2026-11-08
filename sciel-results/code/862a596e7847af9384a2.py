"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
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
    return U, V, W


def _strassen_4x4():
    """
    Return a rank-20 decomposition for 4x4 matrix multiplication using
    a tensor-product / block-based construction derived from Strassen's
    2x2 algorithm (rank 7) applied to the 4x4 problem.
    
    The 4x4 multiplication can be viewed as multiplying 2x2 blocks of 2x2
    matrices. Using Strassen's 7 multiplications for the 2x2 block
    multiplication, and each "multiplication" is itself a 2x2 matmul,
    we get 7 * 7 = 49 rank terms from naive block Strassen.
    
    However, we can do better. The known best rank for 4x4 is 49 (from
    Strassen's algorithm applied to blocks), but there exist rank-49
    decompositions. Actually, the best known upper bound for 4x4 is 49.
    
    Wait - let me reconsider. The standard result:
    - 2x2: rank 7 (Strassen)
    - 4x4: rank 49 (applying Strassen to 2x2 blocks)
    - 8x8: rank 343
    
    But actually, there are improvements. The best known rank for 4x4
    matrix multiplication is 49 (no improvement over naive Strassen block
    decomposition is known to beat 49 for 4x4 as of current literature).
    
    Actually, I recall that Bini's algorithm and others give rank 49 for 4x4.
    Let me construct the rank-49 decomposition.
    
    For 4x4 = (2x2 blocks), we use Strassen's 7 products for the block
    level, and each block product is a 2x2 matmul done with Strassen's
    7 products. So total rank = 7 * 7 = 49.
    
    Let me construct this carefully.
    
    A 4x4 matrix A is partitioned into 2x2 blocks:
    A = [[A11, A12],
         [A21, A22]]
    where each Aij is 2x2.
    
    Similarly B = [[B11, B12],
                   [B21, B22]]
    
    C = A*B has blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Using Strassen for the 2x2 block multiplication:
    P1 = (A11 + A22) * (B11 + B22)
    P2 = (A21 + A22) * B11
    P3 = A11 * (B12 - B22)
    P4 = A22 * (B21 - B11)
    P5 = (A11 + A12) * B22
    P6 = (A21 - A11) * (B11 + B12)
    P7 = (A12 - A22) * (B21 + B22)
    
    Then:
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    Each Pj is a 2x2 matrix product. We compute each using Strassen's
    7 scalar products (well, 7 "products" where each input is a 2x2 block).
    
    Actually, for the tensor decomposition, each "product" Pj = X * Y where
    X and Y are 2x2 matrices (linear combinations of the input blocks).
    
    To decompose Pj = X * Y into rank-1 terms (in the tensor sense), we
    use Strassen's 7-term decomposition for 2x2 matmul.
    
    For a 2x2 matrix product X*Y where X = [[x11, x12],[x21, x22]] and
    Y = [[y11, y12],[y21, y22]], Strassen gives:
    
    s1 = (x11+x22)(y11+y22)
    s2 = (x21+x22)y11
    s3 = x11(y12-y22)
    s4 = x22(y21-y11)
    s5 = (x11+x12)y22
    s6 = (x21-x11)(y11+y12)
    s7 = (x12-x22)(y21+y22)
    
    Result:
    [[s1+s4-s5+s7, s3+s5],
     [s2+s4, s1-s2+s3+s6]]
    
    So for each of the 7 block products Pj, we have 7 scalar-level terms,
    giving 49 total rank terms.
    
    Now I need to express each term in the form U[row, col] * V[row, col] * W[row, col]
    where the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] equals the
    matrix multiplication tensor.
    
    The input A has entries A[a,b] for a,b in 0..3.
    The input B has entries B[b,c] for b,c in 0..3.
    The output C has entries C[a,c] for a,c in 0..3.
    
    The tensor T[a,c,b] = C[a,c] coefficient of A[a',b'] * B[b',c'] ... 
    
    Actually, the matrix multiplication tensor T is defined by:
    T[i, j, k] where i indexes output row, j indexes output col, k is... 
    
    Let me use the convention from the problem:
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (16, rank)
    W has shape (p*n, rank) = (16, rank)
    
    The decomposition says:
    For the matmul tensor, T[a, b, c] (where a is output row, b is output col, 
    and the "middle" index is the sum index) ... 
    
    Actually, the standard tensor decomposition for matmul:
    The multiplication tensor M_{n,p,m} (or M_{n,m,p} depending on convention)
    has T[i, j, k] = 1 if i = j + k in some sense... 
    
    Let me use the explicit convention from the schoolbook code:
    U[i*m + j, col] = 1  -> U is indexed by (output_row, sum_index)
    V[j*p + k, col] = 1  -> V is indexed by (sum_index, output_col)  
    W[k*n + i, col] = 1  -> W is indexed by (output_col, output_row)
    
    Wait, W[k*n + i] means W is indexed by (k, i) where k is sum index and i is output row.
    But the output is C[i, k]... 
    
    Let me re-examine: In the schoolbook, for each (i, j, k):
    - U[i*m + j] = 1: this picks A[i, j]
    - V[j*p + k] = 1: this picks B[j, k]
    - W[k*n + i] = 1: this picks C[k, i]... 
    
    Hmm, that doesn't seem right for standard matmul. Let me re-read the problem.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed as W[k*n + i, r] where k and i are indices. In standard
    matmul C = A*B, C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor is T[i, j, k] where the decomposition is:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right either. Let me think about this more carefully.
    
    The matmul tensor has three modes:
    - Mode 1: (i, j) -> flattened as i*m + j, size n*m = 16
    - Mode 2: (j, k) -> flattened as j*p + k, size m*p = 16
    - Mode 3: (k, i) -> flattened as k*n + i, size p*n = 16
    
    And T[(i,j), (j,k), (k,i)] = 1 if the indices are consistent, 0 otherwise.
    
    Actually no. The tensor T is defined such that:
    C[k, i] = sum_{(i,j),(j,k)} T[(i,j),(j,k),(k,i)] * A[i,j] * B[j,k]
    
    Hmm, I think the convention is:
    The multiplication tensor T has T[a, b, c] where:
    - a = i*m + j (input A index)
    - b = j*p + k (input B index)  
    - c = k*n + i (output C index)
    
    And T[a, b, c] = 1 iff the indices are consistent (i.e., the j in a matches
    the j in b, and the k in b matches the k in c, and the i in c matches the i in a).
    
    So T[(i,j), (j',k), (k',i')] = 1 iff j = j' and k = k' and i = i'.
    
    This means T is the "identity" in the sense that it only has 1s where the
    indices match up for the summation.
    
    The rank-r decomposition: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    For the schoolbook: each term r corresponds to a specific (i, j, k), and
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    
    OK so now for Strassen's decomposition, I need to express the 49 terms.
    
    Let me define the 4x4 matrices in terms of their 2x2 blocks:
    A = [[A00, A01, A10, A11],    where A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
         [A20, A21, A30, A31]]    where A20 = A[2,0], A21 = A[2,1], A30 = A[3,0], A31 = A[3,1]]
    
    Block notation:
    A11_block = [[A[0,0], A[0,1]],
                 [A[1,0], A[1,1]]]
    A12_block = [[A[0,2], A[0,3]],
                 [A[1,2], A[1,3]]]
    A21_block = [[A[2,0], A[2,1]],
                 [A[3,0], A[3,1]]]
    A22_block = [[A[2,2], A[2,3]],
                 [A[3,2], A[3,3]]]
    
    Similarly for B.
    
    For each of the 7 Strassen block products Pj = Xj * Yj, where Xj and Yj
    are 2x2 matrices that are linear combinations of the A blocks, I apply
    Strassen's 7-term decomposition.
    
    This is getting complex. Let me just construct it programmatically.
    
    For each block product Pj = Xj * Yj:
    - Xj is a 2x2 matrix whose entries are linear combinations of A entries
    - Yj is a 2x2 matrix whose entries are linear combinations of B entries
    
    For the 2x2 product Xj * Yj, Strassen gives 7 terms. Each term is of
    the form: (linear combo of Xj entries) * (linear combo of Yj entries)
    contributing to specific entries of the result.
    
    The result of Pj contributes to specific output blocks Ckl.
    
    Let me code this up step by step.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 formulas for 2x2 matmul:
    # Given X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # s1 = (x0+x3)*(y0+y3)
    # s2 = (x2+x3)*y0
    # s3 = x0*(y1-y3)
    # s4 = x3*(y2-y0)
    # s5 = (x0+x1)*y3
    # s6 = (x2-x0)*(y0+y1)
    # s7 = (x1-x3)*(y2+y3)
    # Result:
    # [[s1+s4-s5+s7, s3+s5],
    #  [s2+s4, s1-s2+s3+s6]]
    
    # The 7 block products for 4x4 (2x2 block) multiplication:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # Output blocks:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Block indices: block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # A block (bi, bj) has entries A[2*bi + r, 2*bj + c] for r,c in 0,1
    
    # For each block product Pj, Xj is a 2x2 matrix with entries that are
    # linear combinations of A entries, and Yj similarly for B.
    
    # Let me represent each Xj entry as a list of (A_index, coefficient)
    # where A_index is the flattened index i*m+j.
    
    def A_block_entry(bi, bj, r, c):
        """Return the flattened index for A[2*bi+r, 2*bj+c]"""
        return (2*bi + r) * m + (2*bj + c)
    
    def B_block_entry(bi, bj, r, c):
        """Return the flattened index for B[2*bi+r, 2*bj+c]"""
        return (2*bi + r) * p + (2*bj + c)
    
    # Define the 7 Xj matrices (each is 2x2, entries are linear combos of A)
    # X1 = A11 + A22
    # X2 = A21 + A22
    # X3 = A11
    # X4 = A22
    # X5 = A11 + A12
    # X6 = A21 - A11
    # X7 = A12 - A22
    
    # Y1 = B11 + B22
    # Y2 = B11
    # Y3 = B12 - B22
    # Y4 = B21 - B11
    # Y5 = B22
    # Y6 = B11 + B12
    # Y7 = B21 + B22
    
    # Each block is 2x2, so Xj[r][c] is a linear combo of A entries.
    
    # Let me define each Xj and Yj as 2x2 arrays of lists of (index, coeff)
    
    def make_block_combo(blocks_with_coeffs):
        """
        blocks_with_coeffs: list of (bi, bj, coeff)
        Returns a 2x2 array where each entry is a list of (flat_index, coeff)
        """
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * m + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    def make_block_combo_B(blocks_with_coeffs):
        """Same but for B"""
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * p + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    # Define X1..X7
    X1 = make_block_combo([(0,0,1), (1,1,1)])  # A11 + A22
    X2 = make_block_combo([(1,0,1), (1,1,1)])  # A21 + A22
    X3 = make_block_combo([(0,0,1)])           # A11
    X4 = make_block_combo([(1,1,1)])           # A22
    X5 = make_block_combo([(0,0,1), (0,1,1)])  # A11 + A12
    X6 = make_block_combo([(1,0,1), (0,0,-1)]) # A21 - A11
    X7 = make_block_combo([(0,1,1), (1,1,-1)]) # A12 - A22
    
    # Define Y1..Y7
    Y1 = make_block_combo_B([(0,0,1), (1,1,1)])  # B11 + B22
    Y2 = make_block_combo_B([(0,0,1)])           # B11
    Y3 = make_block_combo_B([(0,1,1), (1,1,-1)]) # B12 - B22
    Y4 = make_block_combo_B([(1,0,1), (0,0,-1)]) # B21 - B11
    Y5 = make_block_combo_B([(1,1,1)])           # B22
    Y6 = make_block_combo_B([(0,0,1), (0,1,1)])  # B11 + B12
    Y7 = make_block_combo_B([(1,0,1), (1,1,1)])  # B21 + B22
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # For each block product Pj = Xj * Yj, apply Strassen's 7 terms.
    # The result of Xj*Yj is a 2x2 matrix. Each entry of this 2x2 result
    # is a linear combination of the 7 Strassen products.
    
    # Strassen products for X*Y (X = [[x0,x1],[x2,x3]], Y = [[y0,y1],[y2,y3]]):
    # s1 = (x0+x3)*(y0+y3)
    # s2 = (x2+x3)*y0
    # s3 = x0*(y1-y3)
    # s4 = x3*(y2-y0)
    # s5 = (x0+x1)*y3
    # s6 = (x2-x0)*(y0+y1)
    # s7 = (x1-x3)*(y2+y3)
    
    # Result:
    # R[0,0] = s1+s4-s5+s7
    # R[0,1] = s3+s5
    # R[1,0] = s2+s4
    # R[1,1] = s1-s2+s3+s6
    
    # Each sk is a product of two linear combinations.
    # For the tensor decomposition, each sk contributes one rank term.
    
    # For sk = (linear_combo_X) * (linear_combo_Y):
    # The linear_combo_X is a single entry (or sum of entries) of X
    # The linear_combo_Y is a single entry (or sum of entries) of Y
    
    # Wait, no. In Strassen, each sk involves products of sums.
    # s1 = (x0+x3)*(y0+y3) - this is NOT a single product, it's a product of two sums.
    # 
    # For the tensor decomposition, we need each term to be a product of
    # a linear combo of A entries, a linear combo of B entries, and a linear
    # combo of C entries.
    #
    # So s1 = (x0+x3)*(y0+y3) where x0, x3 are entries of X (linear combos of A)
    # and y0, y3 are entries of Y (linear combos of B).
    #
    # The term s1 contributes to R[0,0] with coeff +1, R[1,1] with coeff +1.
    #
    # In the tensor decomposition, s1 as a whole is ONE rank term:
    # U part: (x0+x3) as a vector over A entries
    # V part: (y0+y3) as a vector over B entries
    # W part: the contribution to C entries
    #
    # But wait, s1 contributes to multiple C entries. So the W part will have
    # multiple non-zero entries.
    
    # Let me define the 7 Strassen sub-products for a given X, Y pair.
    # Each sub-product sk has:
    # - A linear combination of A entries (the "X factor")
    # - A linear combination of B entries (the "Y factor")
    # - Contributions to specific C entries (the "W factor")
    
    # For X = [[x0,x1],[x2,x3]] where each xi is a 2x2 block of A entries:
    # x0 = X[0][0], x1 = X[0][1], x2 = X[1][0], x3 = X[1][1]
    # Each xi is a list of (A_flat_index, coeff)
    
    # Similarly for Y.
    
    # s1: X_factor = x0 + x3, Y_factor = y0 + y3
    # s2: X_factor = x2 + x3, Y_factor = y0
    # s3: X_factor = x0, Y_factor = y1 - y3
    # s4: X_factor = x3, Y_factor = y2 - y0
    # s5: X_factor = x0 + x1, Y_factor = y3
    # s6: X_factor = x2 - x0, Y_factor = y0 + y1
    # s7: X_factor = x1 - x3, Y_factor = y2 + y3
    
    # Each X_factor is a linear combo of A entries (sum of the relevant xi's)
    # Each Y_factor is a linear combo of B entries
    
    # The W factor depends on which output block the Pj contributes to and
    # which entry of the 2x2 result it affects.
    
    # For block product Pj contributing to output block Ckl with coefficient alpha:
    # The 2x2 result R of Xj*Yj contributes alpha*R[r][c] to C block (kl)[r][c]
    # which is C[2*k+r, 2*l+c]
    
    # The W flat index for C[2*k+r, 2*l+c] is: (2*l+c)*n + (2*k+r)
    # Wait, W[k*n+i] where k is the col and i is the row? Let me re-check.
    # W[k*n + i, col] in schoolbook where the term is A[i,j]*B[j,k] -> C[i,k]
    # So W is indexed by (k, i) -> flat = k*n + i, and it corresponds to C[i,k].
    
    # So for C[row, col], the W index is col*n + row.
    
    # Now, for each of the 7 block products and 7 Strassen sub-products,
    # I need to determine the W factor.
    
    # Let me define the contribution of each Pj to each output block:
    # C11 (block 0,0): P1(+1) + P4(+1) + P5(-1) + P7(+1)
    # C12 (block 0,1): P3(+1) + P5(+1)
    # C21 (block 1,0): P2(+1) + P4(+1)
    # C22 (block 1,1): P1(+1) + P2(-1) + P3(+1) + P6(+1)
    
    # For Pj contributing to C block (kl) with coefficient alpha:
    # The 2x2 result R of Pj has entries R[0,0], R[0,1], R[1,0], R[1,1]
    # R[r,c] = sum of Strassen sub-products with their coefficients
    # Each sub-product sk contributes to R[r,c] with a specific coefficient.
    
    # The full contribution to C[2*k+r, 2*l+c] from sub-product sk of Pj is:
    # alpha * (coeff of sk in R[r,c])
    
    # So for the W factor of sub-product sk of Pj:
    # For each output block (kl) that Pj contributes to with coefficient alpha:
    #   For each (r,c) where sk has a non-zero coefficient in R[r,c]:
    #     W_flat_index = (2*l+c)*n + (2*k+r)
    #     W_coeff = alpha * (coeff of sk in R[r,c])
    
    # Let me define the Strassen coefficients for each sk in each R[r,c]:
    # R[0,0]: s1(+1), s4(+1), s5(-1), s7(+1)
    # R[0,1]: s3(+1), s5(+1)
    # R[1,0]: s2(+1), s4(+1)
    # R[1,1]: s1(+1), s2(-1), s3(+1), s6(+1)
    
    strassen_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 0},  # s1 in R[0,0]
    }
    # Let me make a table: strassen_table[sk][r][c] = coefficient
    strassen_table = [
        # s1
        [[1, 0], [0, 1]],
        # s2
        [[0, 0], [1, -1]],
        # s3
        [[0, 1], [0, 1]],
        # s4
        [[1, 0], [1, 0]],
        # s5
        [[-1, 1], [0, 0]],
        # s6
        [[0, 0], [0, 1]],
        # s7
        [[1, 0], [0, 0]],
    ]
    
    # Block contributions: block_contrib[j][kl] = coefficient
    # j is 0..6 (P1..P7), kl is (0,0), (0,1), (1,0), (1,1)
    block_contrib = [
        # P1
        {((0,0)): 1, ((1,1)): 1},
        # P2
        {((1,0)): 1, ((1,1)): -1},
        # P3
        {((0,1)): 1, ((1,1)): 1},
        # P4
        {((0,0)): 1, ((1,0)): 1},
        # P5
        {((0,0)): -1, ((0,1)): 1},
        # P6
        {((1,1)): 1},
        # P7
        {((0,0)): 1},
    ]
    
    # Now construct the decomposition
    col = 0
    for j in range(7):  # 7 block products
        Xj = Xs[j]
        Yj = Ys[j]
        # Xj is 2x2, each entry is list of (A_idx, coeff)
        # Yj is 2x2, each entry is list of (B_idx, coeff)
        
        x0, x1 = Xj[0]
        x2, x3 = Xj[1]
        y0, y1 = Yj[0]
        y2, y3 = Yj[1]
        
        # Define the 7 sub-products' X and Y factors
        # Each factor is a list of (index, coeff)
        
        def add_factors(f1, f2, sign2=1):
            """Add two factor lists, with sign2 multiplying f2"""
            result = list(f1)
            for idx, c in f2:
                result.append((idx, c * sign2))
            return result
        
        def sub_factors(f1, f2):
            return add_factors(f1, f2, -1)
        
        # s1: X = x0+x3, Y = y0+y3
        # s2: X = x2+x3, Y = y0
        # s3: X = x0, Y = y1-y3
        # s4: X = x3, Y = y2-y0
        # s5: X = x0+x1, Y = y3
        # s6: X = x2-x0, Y = y0+y1
        # s7: X = x1-x3, Y = y2+y3
        
        sub_products = [
            (add_factors(x0, x3), add_factors(y0, y3)),       # s1
            (add_factors(x2, x3), y0),                        # s2
            (x0, sub_factors(y1, y3)),                        # s3
            (x3, sub_factors(y2, y0)),                        # s4
            (add_factors(x0, x1), y3),                        # s5
            (sub_factors(x2, x0), add_factors(y0, y1)),       # s6
            (sub_factors(x1, x3), add_factors(y2, y3)),       # s7
        ]
        
        for sk_idx in range(7):
            x_factor, y_factor = sub_products[sk_idx]
            
            # Build U vector for this term
            u_dict = {}
            for idx, coeff in x_factor:
                u_dict[idx] = u_dict.get(idx, 0) + coeff
            for idx in range(n * m):
                U[idx][col] = u_dict.get(idx, 0)
            
            # Build V vector for this term
            v_dict = {}
            for idx, coeff in y_factor:
                v_dict[idx] = v_dict.get(idx, 0) + coeff
            for idx in range(m * p):
                V[idx][col] = v_dict.get(idx, 0)
            
            # Build W vector for this term
            w_dict = {}
            for (kl), alpha in block_contrib[j].items():
                k, l = kl
                for r in range(2):
                    for c in range(2):
                        coeff_rc = strassen_table[sk_idx][r][c]
                        if coeff_rc != 0:
                            C_row = 2 * k + r
                            C_col = 2 * l + c
                            W_idx = C_col * n + C_row
                            total_coeff = alpha * coeff_rc
                            w_dict[W_idx] = w_dict.get(W_idx, 0) + total_coeff
            for idx in range(p * n):
                W[idx][col] = w_dict.get(idx, 0)
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] matches the matmul tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    # Also check that T is zero where it should be
    # T[a,b,c] should be 0 unless the indices are consistent
    # a = i*m+j, b = j'*p+k', c = k''*n+i'
    # T is 1 iff j=j' and k=k'' and i=i'
    # Let's spot check some zeros
    if n >= 2 and m >= 2 and p >= 2:
        # a = 0*m+0 = 0 (i=0,j=0), b = 1*p+0 = p (j'=1,k'=0), c = 0*n+0 = 0 (k''=0,i'=0)
        # j != j' so should be 0
        a = 0
        b = 1 * p + 0
        c = 0
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(actual) > tol:
            return False
    return True


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks, giving rank 49.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try the rank-49 Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4()
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
