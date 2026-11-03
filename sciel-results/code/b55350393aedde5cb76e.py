"""
Strassen-based rank-98 decomposition for 4x4 * 4x8 matrix multiplication.

Strategy:
  - Decompose A (4x4) and B (4x8) into 2x2 blocks of 2x2 sub-matrices.
  - Use the Strassen algorithm (7 multiplications) on the 4x4 * 4x4 case,
    which gives rank 7 for the 2x2 * 2x2 * 2x2 core extended to 4x4.
  - For 4x4 * 4x8, treat B as two 4x4 halves. Apply Strassen to each half.
  - Total rank: 2 * 49 = 98 (since 4x4*4x4 has minimal rank 49, and we have two independent 4x4*4x4 multiplications).

  Actually, let me think more carefully. The known minimal rank for 4x4 matrix multiplication is 49.
  For 4x4 * 4x8, we can view this as multiplying A (4x4) with B1 (4x4) and B2 (4x4) independently,
  giving two 4x4*4x4 products. Each requires rank 49, so total rank = 98.

  We use the explicit Strassen-based construction for 4x4*4x4 (rank 49) and duplicate it for the two halves of B.

  The 4x4*4x4 Strassen decomposition of rank 49 is built from the 2x2*2x2*2x2 Strassen (rank 7)
  applied recursively. For 4x4 = 2x(2x2), we have a 2x2 grid of 2x2 blocks, giving 4x4 Strassen
  with 7^2 = 49 multiplications.

  Let's construct the 4x4*4x4 rank-49 decomposition explicitly using the recursive Strassen formula.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]

DEADLINE = 150.0  # seconds, with safety margin from 120s evaluator limit

def _strassen_4x4_4x4():
    """
    Return (U4, V4, W4) for 4x4 * 4x4 matrix multiplication with rank 49.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    where row indices are (i*m+j) for U/V, (k*n+i) for W.
    Here n=m=p=4, so U row = i*4+j, V row = j*4+k, W row = k*4+i.
    """
    n = m = p = 4
    rank = 49
    U4 = [[0] * rank for _ in range(n * m)]
    V4 = [[0] * rank for _ in range(m * p)]
    W4 = [[0] * rank for _ in range(p * n)]
    
    # The 4x4 Strassen is built from a 2x2 grid of 2x2 blocks.
    # At the top level, we have 2x2 blocks of size 2x2.
    # Strassen for 2x2 blocks: 7 products M1..M7
    # Each M is a 2x2 * 2x2 = 2x2 product, which itself uses Strassen rank 7.
    # So total 7 * 7 = 49 products, each of rank 1 (scalar 2x2*2x2*2x2).
    
    # Block structure for 4x4:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Strassen products at top level (each is a 2x2 matrix product):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each Mx is a 2x2 matrix, computed as a 2x2*2x2*2x2 product of rank 7.
    # The 2x2 Strassen:
    # For matrices a (2x2), b (2x2), c = a*b:
    # a = [[a00, a01], [a10, a11]], b = [[b00, b01], [b10, b11]]
    # s1 = a10 + a11, s2 = a00 + a01, s3 = a00 - a10, s4 = a01 - a11
    # s5 = b00 + b01, s6 = b10 + b11, s7 = b00 - b10, s8 = b01 - b11
    # p1 = a00 * (b00 + b11)  -> 2x2*2x2*2x2 rank 1? No, each scalar product is rank 1.
    # Actually for 2x2*2x2, the 7 Strassen products are scalar multiplications of linear combinations.
    # p1 = (a00) * (b00 + b11)  -- but these are 2x2 matrices, so each "product" is a 2x2*2x2 = 16 terms, rank 7 via Strassen.
    
    # I need to be more precise. Let me think of this as a tensor decomposition.
    # The 2x2*2x2*2x2 Strassen rank-7 decomposition:
    # For scalar entries, the 7 terms are:
    # term1: a00*(b00+b11)
    # term2: (a10+a11)*b00
    # term3: a00*(b01-b11)
    # term4: a11*(b10-b00)
    # term5: (a00+a01)*b11
    # term6: (a10-a00)*(b00+b01)
    # term7: (a01-a11)*(b10+b11)
    
    # For 2x2 matrices, each "scalar" is a 2x2 matrix, and each product is a 2x2*2x2 product.
    # We use Strassen again for each of these, giving 7 sub-products per top-level product.
    
    # This is getting complex. Let me use a different approach: directly enumerate the 49 terms
    # based on the known structure.
    
    # Actually, let me use a cleaner recursive construction.
    # I'll represent the 4x4 Strassen as follows:
    # The 4x4 matrix is divided into four 2x2 quadrants.
    # We apply Strassen at the 2x2-block level (7 products of 2x2*2x2).
    # Each 2x2*2x2 product is further decomposed using Strassen (7 products of 1x1*1x1).
    
    # Let me define the 2x2 Strassen terms explicitly for scalar matrices, then lift to 2x2 blocks.
    
    # For a 2x2 * 2x2 product with entries:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # The 7 Strassen terms (each is a product of two linear combinations):
    # T1: a00 * (b00 + b11)
    # T2: (a10 + a11) * b00
    # T3: a00 * (b01 - b11)
    # T4: a11 * (b10 - b00)
    # T5: (a00 + a01) * b11
    # T6: (a10 - a00) * (b00 + b01)
    # T7: (a01 - a11) * (b10 + b11)
    
    # For 2x2 blocks, each "entry" is a 2x2 matrix, and each term is a 2x2*2x2 product.
    # We decompose each 2x2*2x2 product using the same 7-term Strassen.
    
    # So we have 7 * 7 = 49 scalar-level terms.
    
    # Let me define the top-level Strassen coefficients and the bottom-level Strassen coefficients.
    
    # Top-level (2x2 blocks of 2x2 matrices):
    # Block indices: 00, 01, 10, 11 for A and B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each Mk, it's a product of two 2x2 matrices. Each 2x2 matrix is a linear combination
    # of 2x2 blocks of A or B. Each such product is decomposed into 7 scalar-level terms.
    
    # Let me define the structure more carefully.
    # A block Aij has entries A[i*2:r, j*2:c] for r in {i*2, i*2+1}, c in {j*2, j*2+1}.
    # In our indexing, A row = i*4+j for the 4x4 matrix, so A[i][j] is at row i*4+j.
    # For 2x2 blocks: block (bi, bj) covers rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1.
    
    # This is getting very involved. Let me try a different, more direct approach.
    
    # I'll use the fact that the 4x4*4x4 rank-49 decomposition can be constructed as:
    # The tensor is the Kronecker product of the 2x2 Strassen tensor with itself.
    # T_4x4 = T_2x2 ⊗ T_2x2
    # where T_2x2 is the rank-7 Strassen tensor for 2x2*2x2*2x2.
    
    # The Strassen tensor for 2x2*2x2*2x2 has 7 terms.
    # The Kronecker product gives 7*7 = 49 terms.
    
    # Let me define the 2x2 Strassen terms with explicit coefficients.
    # For 2x2*2x2: A (2x2), B (2x2), C = A*B (2x2)
    # U rows: (i,j) for i,j in {0,1} -> 4 rows, indexed 0..3
    # V rows: (j,k) for j,k in {0,1} -> 4 rows, indexed 0..3
    # W rows: (k,i) for k,i in {0,1} -> 4 rows, indexed 0..3
    
    # The 7 Strassen terms for 2x2:
    # Term 1: (A00 + A11) * (B00 + B11) -> contributes to C
    # Term 2: (A10 + A11) * B00
    # Term 3: A00 * (B01 - B11)
    # Term 4: A11 * (B10 - B00)
    # Term 5: (A00 + A01) * B11
    # Term 6: (A10 - A00) * (B00 + B01)
    # Term 7: (A01 - A11) * (B10 + B11)
    
    # For each term, the U, V, W factors are:
    # U: coefficients on A entries
    # V: coefficients on B entries
    # W: coefficients on C entries
    
    # Let me define the 2x2 Strassen factors:
    # U is 4x7, V is 4x7, W is 4x7
    
    # A entries: A00=row0, A01=row1, A10=row2, A11=row3
    # B entries: B00=row0, B01=row1, B10=row2, B11=row3
    # C entries: C00=row0, C01=row1, C10=row2, C11=row3
    # But W row index is k*n+i = k*2+i for 2x2, so:
    # C[k][i]: k=0,i=0 -> row 0; k=0,i=1 -> row 1; k=1,i=0 -> row 2; k=1,i=1 -> row 3
    
    # Strassen terms (using standard formulas):
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Wait, let me use the standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # For the 2x2 Strassen tensor:
    # Term 1 (M1): U = [1, 0, 0, 1] (A00+A11), V = [1, 0, 0, 1] (B00+B11), W = [1, 0, 0, 1] (C00+C11... wait)
    # Actually W for M1: M1 contributes +1 to C00 and +1 to C11, so W = [1, 0, 0, 1] in (k,i) order
    # C00: k=0,i=0 -> row 0; C11: k=1,i=1 -> row 3. So W = [1, 0, 0, 1]
    
    # Let me build U2, V2, W2 for 2x2 Strassen:
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    U2[0][0] = 1  # A00
    U2[3][0] = 1  # A11
    V2[0][0] = 1  # B00
    V2[3][0] = 1  # B11
    W2[0][0] = 1  # C00
    W2[3][0] = 1  # C11
    
    # Term 2: M2 = (A10+A11)*B00
    U2[2][1] = 1  # A10
    U2[3][1] = 1  # A11
    V2[0][1] = 1  # B00
    W2[2][1] = 1  # C10 (k=1,i=0)
    W2[3][1] = 1  # C11 (k=1,i=1)
    
    # Term 3: M3 = A00*(B01-B11)
    U2[0][2] = 1  # A00
    V2[1][2] = 1  # B01
    V2[3][2] = -1 # B11
    W2[1][2] = 1  # C01 (k=0,i=1)
    
    # Term 4: M4 = A11*(B10-B00)
    U2[3][3] = 1  # A11
    V2[2][3] = 1  # B10
    V2[0][3] = -1 # B00
    W2[2][3] = 1  # C10
    W2[3][3] = 1  # C11
    
    # Term 5: M5 = (A00+A01)*B11
    U2[0][4] = 1  # A00
    U2[1][4] = 1  # A01
    V2[3][4] = 1  # B11
    W2[1][4] = 1  # C01
    W2[3][4] = 1  # C11
    
    # Term 6: M6 = (A10-A00)*(B00+B01)
    U2[2][5] = 1  # A10
    U2[0][5] = -1 # A00
    V2[0][5] = 1  # B00
    V2[1][5] = 1  # B01
    W2[0][5] = -1 # C00
    W2[2][5] = 1  # C10
    
    # Term 7: M7 = (A01-A11)*(B10+B11)
    U2[1][6] = 1  # A01
    U2[3][6] = -1 # A11
    V2[2][6] = 1  # B10
    V2[3][6] = 1  # B11
    W2[0][6] = 1  # C00
    W2[3][6] = -1 # C11
    
    return U2, V2, W2

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # For 4x4 * 4x8:
    # A is 4x4, B is 4x8, C is 4x8.
    # Split B into B_left (4x4) and B_right (4x4).
    # C = A * B_left concatenated with A * B_right.
    # Each A * B_half is a 4x4 * 4x4 product, rank 49.
    # Total rank = 98.
    
    # Get the 4x4*4x4 rank-49 decomposition via Kronecker product of 2x2 Strassen.
    U2, V2, W2 = _strassen_4x4_4x4()
    
    # The 4x4*4x4 decomposition is the Kronecker product of U2 with itself (for A), etc.
    # Actually, for the Kronecker product of two 2x2 Strassen tensors:
    # The 4x4 matrix A is divided into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # The Kronecker product means:
    # For the 4x4 U matrix (16 rows x 49 cols):
    # Row index: (block_i, block_j, entry_r, entry_c) where block_i,j in {0,1}, entry_r,c in {0,1}
    # In flat index: row = (block_i * 2 + entry_r) * 4 + (block_j * 2 + entry_c) = block_i*8 + entry_r*4 + block_j*2 + entry_c
    
    # Wait, I need to be careful with the indexing.
    # For 4x4 matrix, row = i*4 + j where i,j in 0..3.
    # Block structure: block (bi, bj) covers rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    # So entry (i,j) belongs to block (i//2, j//2) with local coords (i%2, j%2).
    
    # For the Kronecker product of two 2x2 Strassen:
    # The 4x4 U matrix has 16 rows and 49 cols.
    # Col index: (t1, t2) where t1,t2 in 0..6 (the two Strassen term indices).
    # Flat col = t1 * 7 + t2.
    
    # U[row][col] = U2[block_row_entry][t1] * U2[entry_row][t2]
    # where block_row_entry is the row index in U2 corresponding to the block-level entry,
    # and entry_row is the row index in U2 corresponding to the entry-level.
    
    # Hmm, this is the key insight for the Kronecker product of tensor decompositions.
    # If T = T1 ⊗ T2, then the factors are:
    # U_T[i1,i2][t1,t2] = U_T1[i1][t1] * U_T2[i2][t2]
    # V_T[j1,j2][t1,t2] = V_T1[j1][t1] * V_T2[j2][t2]
    # W_T[k1,k2][t1,t2] = W_T1[k1][t1] * W_T2[k2][t2]
    
    # For 4x4*4x4:
    # A is 4x4, indexed by (i,j) where i,j in 0..3.
    # We can write i = 2*i1 + i2, j = 2*j1 + j2 where i1,j1,i2,j2 in {0,1}.
    # Similarly for B: (j,k) -> j = 2*j1+j2, k = 2*k1+k2.
    # For C: (k,i) -> k = 2*k1+k2, i = 2*i1+i2.
    
    # The 4x4*4x4 tensor T4[i,j,k] = sum_{j} A[i,j]*B[j,k]
    # = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    
    # This is NOT simply the Kronecker product of two 2x2 tensors because the summation index j
    # is split into j1 and j2, and they appear in both A and B.
    
    # Actually, the correct way: the 4x4*4x4 matrix multiplication tensor is the
    # "tensor product" in a specific sense. Let me think again.
    
    # The 4x4*4x4 multiplication can be viewed as:
    # C[i,j] (using C row=i, col=j... wait, let me use consistent notation)
    # Let A be 4x4, B be 4x4, C = A*B be 4x4.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2.
    # C[2*i1+i2, 2*k1+k2] = sum_{j1=0}^{1} sum_{j2=0}^{1} A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    
    # This is a 2x2 grid of 2x2*2x2*2x2 sub-problems, but with shared summation indices.
    # It's NOT a simple Kronecker product.
    
    # The correct construction uses the Strassen algorithm at the block level.
    # At the block level, we have a 2x2 grid of 2x2 blocks.
    # We apply Strassen to the 2x2-block matrix multiplication, which requires 7 block products.
    # Each block product is a 2x2*2x2 matrix multiplication, which we decompose using Strassen (rank 7).
    # Total: 7 * 7 = 49.
    
    # Let me implement this properly.
    
    # Block-level Strassen for 2x2 grid of 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = A*B
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Where M1..M7 are as defined above, each being a 2x2*2x2 product.
    
    # For each Mt, it's a product of two 2x2 matrices. Each 2x2 matrix is a linear combination
    # of 2x2 blocks. We decompose each Mt using the 2x2 Strassen (U2, V2, W2).
    
    # For Mt = X * Y where X and Y are 2x2 matrices (each a linear combination of blocks):
    # X is determined by its coefficients on A00, A01, A10, A11.
    # Y is determined by its coefficients on B00, B01, B10, B11.
    # The product Mt = X*Y is a 2x2 matrix, which is the sum of contributions to C blocks.
    
    # For the decomposition, for each Mt:
    # - The U part: for each entry in X (which is a 2x2 matrix), we have the Strassen U2 factors.
    #   But X is a linear combination of blocks. Each block is a 2x2 matrix.
    #   So X = sum_{a in {00,01,10,11}} x_a * A_a where x_a are coefficients.
    #   Similarly Y = sum_{b in {00,01,10,11}} y_b * B_b.
    #   
    #   The product Mt = X*Y = (sum_a x_a * A_a) * (sum_b y_b * B_b)
    #   = sum_a sum_b x_a * y_b * (A_a * B_b)
    #   
    #   Each A_a * B_b is a 2x2*2x2 product, decomposed by U2, V2, W2 (rank 7).
    #   So Mt has 4*4 = 16 sub-products, each of rank 7, total 16*7 = 112? That's too many.
    
    # I think I'm overcomplicating this. Let me use a different approach.
    
    # Actually, the standard way to get the 4x4 rank-49 decomposition is:
    # Use the Strassen algorithm recursively. The 4x4*4x4 multiplication uses 7 multiplications
    # of 2x2*2x2 matrices (at the block level). Each 2x2*2x2 multiplication uses 7 scalar multiplications.
    # Total: 49 scalar multiplications.
    
    # The key is that each "block multiplication" Mt = X*Y where X and Y are 2x2 matrices
    # (not 2x2 blocks of the original 4x4, but actual 2x2 matrices formed from linear combinations).
    
    # Wait, I think the confusion is about what "block" means. Let me restart with a clearer formulation.
    
    # For 4x4 * 4x4:
    # Divide A into four 2x2 sub-matrices: A00, A01, A10, A11 (top-left, top-right, bottom-left, bottom-right).
    # Similarly B into B00, B01, B10, B11.
    # C = A*B is 4x4, divided into C00, C01, C10, C11.
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mt is a 2x2 matrix (product of two 2x2 matrices).
    # Each Mt is computed as a 2x2*2x2 matrix product, which we decompose using Strassen (rank 7).
    
    # So for each Mt, we need to express it as a sum of 7 rank-1 terms.
    # The 7 rank-1 terms for Mt = X*Y (where X and Y are 2x2 matrices) are given by U2, V2, W2.
    
    # But X and Y are linear combinations of the original 4x4 sub-matrices.
    # For example, M1 = (A00+A11)*(B00+B11).
    # X = A00+A11 (a 2x2 matrix), Y = B00+B11 (a 2x2 matrix).
    # M1 = X*Y.
    # Using Strassen for X*Y:
    # M1_1 = X00 * (Y00+Y11) where X00 is the (0,0) entry of X, etc.
    # But X00 = A00_00 + A11_00 (the (0,0) entries of A00 and A11).
    # Similarly Y00 = B00_00 + B11_00.
    # So M1_1 = (A00_00 + A11_00) * (B00_00 + B11_00).
    # This is a scalar product (rank 1 term in the full 4x4*4x4 decomposition).
    
    # So each Mt contributes 7 scalar-level rank-1 terms, and we have 7 Mt's, giving 49 total.
    
    # Let me implement this.
    
    # First, define the block-level Strassen coefficients.
    # For Mt, X = sum_{a} x_a * A_a, Y = sum_{b} y_b * B_b.
    # x_a and y_b are coefficients in {0, 1, -1}.
    
    # M1: X = A00 + A11, Y = B00 + B11
    #   x = {00:1, 01:0, 10:0, 11:1}, y = {00:1, 01:0, 10:0, 11:1}
    # M2: X = A10 + A11, Y = B00
    #   x = {00:0, 01:0, 10:1, 11:1}, y = {00:1, 01:0, 10:0, 11:0}
    # M3: X = A00, Y = B01 - B11
    #   x = {00:1, 01:0, 10:0, 11:0}, y = {00:0, 01:1, 10:0, 11:-1}
    # M4: X = A11, Y = B10 - B00
    #   x = {00:0, 01:0, 10:0, 11:1}, y = {00:-1, 01:0, 10:1, 11:0}
    # M5: X = A00 + A01, Y = B11
    #   x = {00:1, 01:1, 10:0, 11:0}, y = {00:0, 01:0, 10:0, 11:1}
    # M6: X = A10 - A00, Y = B00 + B01
    #   x = {00:-1, 01:0, 10:1, 11:0}, y = {00:1, 01:1, 10:0, 11:0}
    # M7: X = A01 - A11, Y = B10 + B11
    #   x = {00:0, 01:1, 10:0, 11:-1}, y = {00:0, 01:0, 10:1, 11:1}
    
    # Block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    # In flat index for 4x4: block (bi,bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # Entry (i,j) in A: block (i//2, j//2), local (i%2, j%2).
    
    # For each Mt and each of its 7 Strassen sub-terms:
    # The sub-term is: (sum of A entries) * (sum of B entries) * (contribution to C entries)
    # Using U2, V2, W2 for the 2x2 Strassen.
    
    # Let me define the block-level coefficients as arrays.
    # Blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # Similarly for B and C.
    
    M_coeffs = [
        # (x_coeffs, y_coeffs) where x_coeffs = [x00, x01, x10, x11], y_coeffs = [y00, y01, y10, y11]
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # M7
    ]
    
    # C block coefficients for each Mt:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    C_coeffs = [
        # (c00, c01, c10, c11) for each Mt
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 1, 0, -1], # M5
        [-1, 0, 1, 1], # M6
        [1, 0, 0, -1], # M7
    ]
    
    # Now build U4, V4, W4 for 4x4*4x4 (rank 49).
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Block to entry mapping:
    # Block b (0..3) -> (bi, bj) where bi = b//2, bj = b%2
    # Entry (i,j) in 4x4 -> block (i//2, j//2), local (i%2, j%2)
    # Local entry (li, lj) in 2x2 block -> U2 row index = li*2 + lj
    
    def block_to_bi_bj(b):
        return b // 2, b % 2
    
    def entry_to_block_local(i, j):
        return i // 2, j // 2, i % 2, j % 2
    
    for mt in range(7):
        x_coeffs, y_coeffs = M_coeffs[mt]
        c_coeffs = C_coeffs[mt]
        
        for t2 in range(7):  # Strassen sub-term index for the 2x2*2x2 product
            col = mt * 7 + t2
            
            # U part: for each entry (i,j) in A (4x4):
            # U4[i*4+j][col] = sum over blocks a: x_coeffs[a] * U2[local_i*2+local_j][t2]
            # where (i,j) is in block a and local coords are (li, lj)
            for i in range(4):
                for j in range(4):
                    bi, bj, li, lj = entry_to_block_local(i, j)
                    a = bi * 2 + bj  # block index
                    U4[i*4+j][col] += x_coeffs[a] * U2[li*2+lj][t2]
            
            # V part: for each entry (j,k) in B (4x4):
            # V4[j*4+k][col] = sum over blocks b: y_coeffs[b] * V2[local_j*2+local_k][t2]
            for j in range(4):
                for k in range(4):
                    bj, bk, lj, lk = entry_to_block_local(j, k)
                    b = bj * 2 + bk  # block index
                    V4[j*4+k][col] += y_coeffs[b] * V2[lj*2+lk][t2]
            
            # W part: for each entry (k,i) in C (4x4):
            # W4[k*4+i][col] = sum over blocks c: c_coeffs[c] * W2[local_k*2+local_i][t2]
            # where c_coeffs[c] is the coefficient of Mt in C_c
            for k in range(4):
                for i in range(4):
                    ck, ci, lk, li = entry_to_block_local(k, i)
                    c = ck * 2 + ci  # block index
                    W4[k*4+i][col] += c_coeffs[c] * W2[lk*2+li][t2]
    
    # Now build the full 4x4*4x8 decomposition (rank 98).
    # B is 4x8, split into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # C is 4x8, split into C_left (4x4, cols 0-3) and C_right (4x4, cols 4-7).
    # C_left = A * B_left, C_right = A * B_right.
    # Each uses the 4x4*4x4 rank-49 decomposition.
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Left half (cols 0-3 of B, cols 0-3 of C):
    for col in range(49):
        for row in range(16):
            U[row][col] = U4[row][col]
        for row in range(16):
            V[row][col] = V4[row][col]
        for row in range(16):
            W[row][col] = W4[row][col]
    
    # Right half (cols 4-7 of B, cols 4-7 of C):
    # B_right[j][k] = B[j][k+4] for k in 0..3
    # C_right[k][i] = C[k][i+4] for k,i in 0..3
    # The decomposition is the same as U4, V4, W4 but with shifted row indices in V and W.
    for col in range(49):
        col_offset = 49 + col
        for row in range(16):
            U[row][col_offset] = U4[row][col]
        for j in range(4):
            for k in range(4):
                # B_right entry (j, k) corresponds to B entry (j, k+4)
                # V row index for B: j*8 + (k+4)
                # V4 row index: j*4 + k
                V[j*8 + (k+4)][col_offset] = V4[j*4 + k][col]
        for k in range(4):
            for i in range(4):
                # C_right entry (k, i) corresponds to C entry (k, i+4)
                # W row index for C: k*4 + (i+4)
                # W4 row index: k*4 + i
                W[k*4 + (i+4)][col_offset] = W4[k*4 + i][col]
    
    # Verify within time budget
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
