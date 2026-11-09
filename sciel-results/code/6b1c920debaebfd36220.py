"""
Matrix multiplication 4x4 x 4x6 using Strassen-based block decomposition.
We split the 4x4 matrix A into 2x2 blocks of 2x2 submatrices, and the 4x6 matrix B
into 2x3 blocks of 2x2 and 2x1 submatrices. However, to keep it simple and correct,
we'll use a recursive approach:

Actually, let's think more carefully. We have A (4x4) * B (4x6) = C (4x6).

We can view this as multiplying a 4x4 matrix by a 4x6 matrix. One clean approach:
- Split A into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
- Split B into two 2x3 blocks vertically: B = [B_left, B_right] where B_left is 4x3 and B_right is 4x3
  Or better: split B into 2x2 and 2x1 blocks? No, 4x6 can be split as 2x3 blocks of 2x2.

Actually, the cleanest approach: treat the 4x6 multiplication as two separate 4x3 multiplications.
For each 4x3 multiplication, we can further decompose.

But the directive says to use Strassen-Winograd 2x2x2 as a building block and leverage
the known optimal rank 18 for 4x4x4.

Let me think about this differently. The rank of M_{4,4,6} is at most the sum of ranks
of M_{4,4,3} + M_{4,4,3} if we split the 6 columns into two groups of 3.

Actually, a better decomposition: 
- 4x4 * 4x6 can be seen as:
  - 4x4 * 4x4 (first 4 columns) + 4x4 * 4x2 (last 2 columns)
  - Rank(4x4*4x4) = 18 (known optimal)
  - Rank(4x4*4x2) <= 18 (can use Strassen-like or just bound by 4*4*2=32, but we can do better)

Actually, let's just implement a correct Strassen-based decomposition.

For 4x4 * 4x6:
Split A (4x4) into 2x2 blocks of 2x2 matrices.
Split B (4x6) into 2x3 blocks of 2x2 matrices (since 6/2=3).

So B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x2.

Then C (4x6) = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x2.

C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C02 = A00*B02 + A01*B12
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11
C12 = A10*B02 + A11*B12

Each Cij is a 2x2 matrix product sum. We can use Strassen for each pair.

For C00: use Strassen on (A00, B00) and (A01, B10) — but Strassen computes one 2x2 product.
We need A00*B00 + A01*B10. This is a sum of two 2x2 products.

We can use the fact that 2x2*2x2 has rank 7 (Strassen). So each product Aij*Bjk costs 7.
We have 6 such products in the 4x4*4x6 (since 2x3 grid of block products, but each is 2x2*2x2).

Wait, actually each Cij = A0j*B0i + A1j*B1i... no let me re-index.

Let me use standard block notation:
A = [[A11, A12], [A21, A22]] (2x2 blocks, each 2x2)
B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks, each 2x2)

C = [[C11, C12, C13], [C21, C22, C23]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C13 = A11*B13 + A12*B23
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22
C23 = A21*B13 + A22*B23

Each Cij is a sum of two 2x2 matrix products. Using Strassen, each 2x2 product uses 7 scalar multiplications.
So total: 6 Cij blocks * 2 products each * 7 = 84 scalar multiplications.

This gives rank 84, which is better than 96 (schoolbook).

Let me implement this Strassen-based decomposition.
"""

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the U, V, W matrices for a Strassen-based decomposition.
    # A is n x m = 4 x 4, B is m x p = 4 x 6, C is n x p = 4 x 6.
    # 
    # Block structure:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each block is 2x2
    # C = [[C11, C12, C13], [C21, C22, C23]] where each block is 2x2
    #
    # For each Cij = A1j*B1i + A2j*B2i (with appropriate indices)
    # We use Strassen's algorithm for each 2x2 product.
    
    # Strassen's 7 products for X*Y where X, Y are 2x2:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y10 - Y00)
    # M4 = X11 * (Y11 - Y10)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    #
    # Result:
    # Z00 = M1 + M4 - M2 + M6
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M5 - M7
    
    # For our case, each Cij = P1 + P2 where P1 = X*Y and P2 = X'*Y'
    # We need to combine the Strassen terms.
    
    # Let's define the 7 Strassen terms for a generic 2x2 product X*Y:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    #
    # The 7 terms are linear combinations of the entries of X and Y.
    # For the tensor decomposition, each rank-1 term corresponds to one scalar multiplication.
    
    # Let's build the decomposition by enumerating all the Strassen terms for each block product.
    
    # First, let's define the Strassen terms as (coeff_x, coeff_y, result_pos, coeff_result)
    # where coeff_x is a 2x2 matrix of coefficients for X, coeff_y is a 2x2 matrix for Y,
    # and the result contributes coeff_result to the Z matrix at position result_pos.
    
    # Actually, let's think of it as: for each of the 7 Strassen terms, we have:
    # term = (sum of X entries with coefficients) * (sum of Y entries with coefficients)
    # and this term contributes to the Z entries with certain coefficients.
    
    # Let's define the 7 Strassen terms explicitly:
    # For X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    
    strassen_terms = [
        # (X coeffs: [x00, x01, x10, x11], Y coeffs: [y00, y01, y10, y11], 
        #  result coeffs: [z00, z01, z10, z11])
        # M1 = (x00 + x11) * (y00 + y11) -> z00 += 1, z11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (x10 + x11) * y00 -> z00 -= 1, z10 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),
        # M3 = x00 * (y10 - y00) -> z01 += 1, z11 -= 1
        ([1, 0, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]),
        # M4 = x11 * (y11 - y10) -> z00 += 1, z10 += 1, z11 -= 1
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 1, -1]),
        # M5 = (x00 + x01) * y11 -> z01 += 1, z11 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        # M6 = (x10 - x00) * (y00 + y01) -> z00 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        # M7 = (x01 - x11) * (y10 + y11) -> z11 -= 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]),
    ]
    
    # Now, for each Cij block, we have two products:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # For each product X*Y, we use 7 Strassen terms.
    # So for each Cij, we have 14 terms (7 for each product).
    # Total: 6 * 14 = 84 terms.
    
    # Let's build U, V, W.
    # U has shape (n*m, rank) = (16, 84)
    # V has shape (m*p, rank) = (24, 84)  
    # W has shape (p*n, rank) = (24, 84)
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index helpers:
    # A[i,j] is at U[i*m + j]
    # B[j,k] is at V[j*p + k]
    # C[i,k] is at W[k*n + i]
    
    # Block indices:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,3]
    # A21: i in [2,3], j in [0,1]
    # A22: i in [2,3], j in [2,3]
    #
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,3]
    # B13: j in [0,1], k in [4,5]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    # B23: j in [2,3], k in [4,5]
    
    # C11: i in [0,1], k in [0,1]
    # C12: i in [0,1], k in [2,3]
    # C13: i in [0,1], k in [4,5]
    # C21: i in [2,3], k in [0,1]
    # C22: i in [2,3], k in [2,3]
    # C23: i in [2,3], k in [4,5]
    
    def get_A_indices(i_base, j_base):
        """Get the 4 indices [i0j0, i0j1, i1j0, i1j1] for A block starting at (i_base, j_base)."""
        return [
            (i_base + 0) * m + (j_base + 0),
            (i_base + 0) * m + (j_base + 1),
            (i_base + 1) * m + (j_base + 0),
            (i_base + 1) * m + (j_base + 1),
        ]
    
    def get_B_indices(j_base, k_base):
        """Get the 4 indices [j0k0, j0k1, j1k0, j1k1] for B block starting at (j_base, k_base)."""
        return [
            (j_base + 0) * p + (k_base + 0),
            (j_base + 0) * p + (k_base + 1),
            (j_base + 1) * p + (k_base + 0),
            (j_base + 1) * p + (k_base + 1),
        ]
    
    def get_C_indices(i_base, k_base):
        """Get the 4 indices [k0i0, k0i1, k1i0, k1i1] for C block starting at (i_base, k_base)."""
        # W is indexed as W[k*n + i]
        return [
            (k_base + 0) * n + (i_base + 0),
            (k_base + 0) * n + (i_base + 1),
            (k_base + 1) * n + (i_base + 0),
            (k_base + 1) * n + (i_base + 1),
        ]
    
    # Define the block products for each Cij:
    # Cij = A_left * B_top + A_right * B_bottom
    # where the blocks are 2x2.
    
    # For C11 (i_base=0, k_base=0): A11*B11 + A12*B21
    #   A11: i_base=0, j_base=0; A12: i_base=0, j_base=2
    #   B11: j_base=0, k_base=0; B21: j_base=2, k_base=0
    block_products = [
        # (C_i_base, C_k_base, A1_i_base, A1_j_base, A2_i_base, A2_j_base, B1_j_base, B1_k_base, B2_j_base, B2_k_base)
        (0, 0, 0, 0, 0, 2, 0, 0, 2, 0),  # C11
        (0, 2, 0, 0, 0, 2, 0, 2, 2, 2),  # C12
        (0, 4, 0, 0, 0, 2, 0, 4, 2, 4),  # C13
        (2, 0, 2, 0, 2, 2, 0, 0, 2, 0),  # C21
        (2, 2, 2, 0, 2, 2, 0, 2, 2, 2),  # C22
        (2, 4, 2, 0, 2, 2, 0, 4, 2, 4),  # C23
    ]
    
    col = 0
    for (ci, ck, a1i, a1j, a2i, a2j, b1j, b1k, b2j, b2k) in block_products:
        # First product: A1 * B1
        A1_indices = get_A_indices(a1i, a1j)
        B1_indices = get_B_indices(b1j, b1k)
        C_indices = get_C_indices(ci, ck)
        
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for idx in range(4):
                U[A1_indices[idx]][col] += x_coeffs[idx]
                V[B1_indices[idx]][col] += y_coeffs[idx]
                W[C_indices[idx]][col] += z_coeffs[idx]
            col += 1
        
        # Second product: A2 * B2
        A2_indices = get_A_indices(a2i, a2j)
        B2_indices = get_B_indices(b2j, b2k)
        C_indices = get_C_indices(ci, ck)
        
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for idx in range(4):
                U[A2_indices[idx]][col] += x_coeffs[idx]
                V[B2_indices[idx]][col] += y_coeffs[idx]
                W[C_indices[idx]][col] += z_coeffs[idx]
            col += 1
    
    return {"U": U, "V": V, "W": W}
