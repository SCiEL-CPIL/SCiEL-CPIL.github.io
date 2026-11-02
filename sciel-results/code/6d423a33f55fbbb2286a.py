"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 x 3x8 matrix multiplication.
    
    Strategy: Split the 3x8 output into a 1x8 block and a 2x8 block.
    - The 1x8 block (1x3 * 3x8) requires rank 3 (schoolbook, since it's just 3 dot products of length 3... wait.
      1x3 * 3x8. Input A is 1x3, B is 3x8. Output is 1x8.
      Each output element C[0, k] = sum_j A[0, j] B[j, k].
      This is a bilinear form in A and B for each k.
      Actually, the tensor is sum_{j,k} A[0,j] B[j,k] C[0,k].
      This can be decomposed with rank 3? No, rank is the number of terms.
      Term j: A[0,j] * B[j,k] * C[0,k]. But C[0,k] depends on k.
      So for fixed j, we have sum_k B[j,k] C[0,k]. This is a dot product of row j of B and C.
      We can treat the "C" factor as having 8 dimensions.
      The tensor T_{a, b, c} where a is index of A (3 dims), b is index of B (24 dims), c is index of C (8 dims).
      T = sum_{j=0}^2 sum_{k=0}^7 e_{j}^{(A)} \otimes (e_{j,k}^{(B)}) \otimes (e_{k}^{(C)}).
      This is a sum of 24 terms. Can we do better?
      The rank of the matrix multiplication tensor for 1xn x nxm is nm.
      So 1x3 x 3x8 has rank 24.
      
      Wait, let's re-evaluate the split.
      A is 2x3. B is 3x8. C is 2x8.
      Split A into A1 (1x3) and A2 (2x3).
      C = A1 B + A2 B.
      A1 B is 1x8. Rank 24? No, 1*3*8 = 24.
      A2 B is 2x8. Rank 2*3*8 = 48 schoolbook.
      Total 72. Worse than 48.
      
      Let's try splitting B columns.
      B = [B1, B2] where B1 is 3x4, B2 is 3x4.
      C = [A B1, A B2].
      A B1 is 2x4. Rank 2*3*4 = 24.
      A B2 is 2x4. Rank 2*3*4 = 24.
      Total 48. Same as schoolbook.
      
      Let's look at the "Measured: 1 -> 1.091" hint.
      "Reduced rank from schoolbook 48 to 44 by splitting m=3 into 1+2 and using Strassen for the 2x2x2 blocks of the 2x2x8 part"
      
      Let's analyze 2x2x2 Strassen.
      2x2 * 2x2 -> 2x2. Rank 7.
      Schoolbook is 8.
      We have 2x3 * 3x8.
      Split m=3 into 1+2.
      A = [A_col0 | A_cols12] (2x1 | 2x2)
      B = [B_row0; B_rows12] (1x8; 2x8)
      C = A_col0 B_row0 + A_cols12 B_rows12.
      
      Term 1: 2x1 * 1x8 -> 2x8.
      Rank of 2x1 * 1x8?
      A is 2x1 (vector a), B is 1x8 (vector b). C = a b^T.
      C[i, j] = a[i] b[j].
      Tensor: sum_{i,j} a[i] b[j] C[i,j].
      This is the outer product tensor. Rank is 2*8 = 16?
      Wait, rank of outer product of u (dim n) and v (dim m) is nm.
      So 2x1 * 1x8 has rank 16.
      
      Term 2: 2x2 * 2x8 -> 2x8.
      A is 2x2, B is 2x8.
      Split B into four 2x2 blocks: B = [B11, B12, B13, B14] (2x2 each).
      C = [C11, C12, C13, C14] (2x2 each).
      C_k = A B_k.
      Each is 2x2 * 2x2. Rank 7 (Strassen).
      Total for Term 2: 4 * 7 = 28.
      
      Total Rank = 16 + 28 = 44.
      This matches the hint! 44 < 48.
      
      So the plan is:
      1. Decompose the 2x1 * 1x8 part with rank 16.
      2. Decompose the 2x2 * 2x8 part using 4 Strassen blocks of rank 7 each.
      3. Combine them.
      
      Let's verify the indices.
      A (2x3): indices (i, j) with i in {0,1}, j in {0,1,2}.
      B (3x8): indices (j, k) with j in {0,1,2}, k in {0..7}.
      C (2x8): indices (i, k) with i in {0,1}, k in {0..7}.
      Output factor W index convention: W[k*n + i, r] where n=2. So W index is 2k + i.
      
      Part 1: 2x1 * 1x8.
      A_part: column 0 of A. Indices (i, 0).
      B_part: row 0 of B. Indices (0, k).
      C_part: all C. Indices (i, k).
      Term (i, k): A[i, 0] * B[0, k] * C[i, k].
      There are 2 * 8 = 16 terms.
      For each term t (0..15):
        i = t // 8
        k = t % 8
        U[2*i + 0, t] = 1  (A index: row i, col 0 -> i*3 + 0 = i*3)
        Wait, U shape is (n*m, rank) = (6, rank).
        A index is i*m + j. Here j=0. So U[i*3 + 0, t] = 1.
        V shape is (m*p, rank) = (24, rank).
        B index is j*p + k. Here j=0. So V[0*8 + k, t] = 1.
        W shape is (p*n, rank) = (16, rank).
        C index is k*n + i. So W[k*2 + i, t] = 1.
        
      Part 2: 2x2 * 2x8.
      A_part: columns 1, 2 of A. Indices (i, 1), (i, 2).
      B_part: rows 1, 2 of B. Indices (1, k), (2, k).
      C_part: all C. Indices (i, k).
      
      We split the 8 columns of B (and C) into 4 blocks of 2 columns.
      Block b (0..3) corresponds to k in {2b, 2b+1}.
      For each block b, we compute C_block = A_sub * B_block.
      A_sub is 2x2 (rows 0,1; cols 1,2).
      B_block is 2x2 (rows 1,2; cols 2b, 2b+1).
      C_block is 2x2 (rows 0,1; cols 2b, 2b+1).
      
      Strassen for 2x2 * 2x2:
      Let A_sub = [[a00, a01], [a10, a11]] corresponding to A[i, j] with i in {0,1}, j in {1,2}.
      So a00 = A[0,1], a01 = A[0,2], a10 = A[1,1], a11 = A[1,2].
      Let B_block = [[b00, b01], [b10, b11]] corresponding to B[j, k] with j in {1,2}, k in {2b, 2b+1}.
      So b00 = B[1, 2b], b01 = B[1, 2b+1], b10 = B[2, 2b], b11 = B[2, 2b+1].
      
      Strassen multipliers (7 terms):
      m1 = (a00 + a11) * (b00 + b11)
      m2 = (a10 + a11) * b00
      m3 = a00 * (b01 - b11)
      m4 = a11 * (b10 - b00)
      m5 = (a00 + a01) * b11
      m6 = (a10 - a00) * (b00 + b01)
      m7 = (a01 - a11) * (b10 + b11)
      
      Results:
      c00 = m1 + m4 - m5 + m7
      c01 = m3 + m5
      c10 = m2 + m4
      c11 = m1 - m2 + m3 + m6
      
      We need to map these to the tensor decomposition.
      Each multiplier m_x is a product of a linear form in A_sub and a linear form in B_block.
      The result C_block entries are linear combinations of m_x.
      
      For each block b, we add 7 terms to the rank.
      Total terms for Part 2 = 4 * 7 = 28.
      Total Rank = 16 + 28 = 44.
      
      Let's construct the matrices U, V, W.
      Rank R = 44.
      U: 6x44
      V: 24x44
      W: 16x44
      
      Initialize with zeros.
      
      Part 1: t = 0..15
      i = t // 8
      k = t % 8
      U[i*3 + 0, t] = 1
      V[0*8 + k, t] = 1
      W[k*2 + i, t] = 1
      
      Part 2:
      For b in 0..3:
        base_col = 16 + 7*b
        # m1: (a00+a11) * (b00+b11)
        # a00 = A[0,1] (idx 0*3+1=1), a11 = A[1,2] (idx 1*3+2=5)
        # b00 = B[1,2b] (idx 1*8+2b), b11 = B[2,2b+1] (idx 2*8+2b+1)
        col = base_col + 0
        U[1, col] = 1
        U[5, col] = 1
        V[8+2*b, col] = 1
        V[16+2*b+1, col] = 1
        # c00 += m1, c11 += m1
        # c00 is C[0, 2b] (idx 2b*2+0 = 4b)
        # c11 is C[1, 2b+1] (idx (2b+1)*2+1 = 4b+3)
        W[4*b, col] = 1
        W[4*b+3, col] = 1
        
        # m2: (a10+a11) * b00
        # a10 = A[1,1] (idx 4), a11 = A[1,2] (idx 5)
        # b00 = B[1,2b] (idx 8+2b)
        col = base_col + 1
        U[4, col] = 1
        U[5, col] = 1
        V[8+2*b, col] = 1
        # c10 += m2
        # c10 is C[1, 2b] (idx 4b+2)
        W[4*b+2, col] = 1
        
        # m3: a00 * (b01-b11)
        # a00 = A[0,1] (idx 1)
        # b01 = B[1,2b+1] (idx 9+2b), b11 = B[2,2b+1] (idx 17+2b)
        col = base_col + 2
        U[1, col] = 1
        V[9+2*b, col] = 1
        V[17+2*b, col] = -1
        # c00 += m3, c11 += m3
        W[4*b, col] = 1
        W[4*b+3, col] = 1
        
        # m4: a11 * (b10-b00)
        # a11 = A[1,2] (idx 5)
        # b10 = B[2,2b] (idx 16+2b), b00 = B[1,2b] (idx 8+2b)
        col = base_col + 3
        U[5, col] = 1
        V[16+2*b, col] = 1
        V[8+2*b, col] = -1
        # c00 += m4, c10 += m4
        W[4*b, col] = 1
        W[4*b+2, col] = 1
        
        # m5: (a00+a01) * b11
        # a00 = A[0,1] (idx 1), a01 = A[0,2] (idx 2)
        # b11 = B[2,2b+1] (idx 17+2b)
        col = base_col + 4
        U[1, col] = 1
        U[2, col] = 1
        V[17+2*b, col] = 1
        # c00 -= m5, c01 += m5
        # c01 is C[0, 2b+1] (idx 4b+1)
        W[4*b, col] = -1
        W[4*b+1, col] = 1
        
        # m6: (a10-a00) * (b00+b01)
        # a10 = A[1,1] (idx 4), a00 = A[0,1] (idx 1)
        # b00 = B[1,2b] (idx 8+2b), b01 = B[1,2b+1] (idx 9+2b)
        col = base_col + 5
        U[4, col] = 1
        U[1, col] = -1
        V[8+2*b, col] = 1
        V[9+2*b, col] = 1
        # c11 += m6
        W[4*b+3, col] = 1
        
        # m7: (a01-a11) * (b10+b11)
        # a01 = A[0,2] (idx 2), a11 = A[1,2] (idx 5)
        # b10 = B[2,2b] (idx 16+2b), b11 = B[2,2b+1] (idx 17+2b)
        col = base_col + 6
        U[2, col] = 1
        U[5, col] = -1
        V[16+2*b, col] = 1
        V[17+2*b, col] = 1
        # c00 += m7
        W[4*b, col] = 1
        
      Return U, V, W as lists of lists.
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1 * 1x8 (rank 16)
    for t in range(16):
        i = t // 8
        k = t % 8
        U[i * m + 0][t] = 1
        V[0 * p + k][t] = 1
        W[k * n + i][t] = 1
        
    # Part 2: 2x2 * 2x8 (4 blocks of Strassen rank 7)
    for b in range(4):
        base_col = 16 + 7 * b
        # m1
        col = base_col + 0
        U[0 * m + 1][col] = 1   # A[0,1]
        U[1 * m + 2][col] = 1   # A[1,2]
        V[1 * p + 2*b][col] = 1 # B[1, 2b]
        V[2 * p + 2*b+1][col] = 1 # B[2, 2b+1]
        W[2*b * n + 0][col] = 1 # C[0, 2b]
        W[(2*b+1) * n + 1][col] = 1 # C[1, 2b+1]
        
        # m2
        col = base_col + 1
        U[1 * m + 1][col] = 1   # A[1,1]
        U[1 * m + 2][col] = 1   # A[1,2]
        V[1 * p + 2*b][col] = 1 # B[1, 2b]
        W[2*b * n + 1][col] = 1 # C[1, 2b]
        
        # m3
        col = base_col + 2
        U[0 * m + 1][col] = 1   # A[0,1]
        V[1 * p + 2*b+1][col] = 1 # B[1, 2b+1]
        V[2 * p + 2*b+1][col] = -1 # B[2, 2b+1]
        W[2*b * n + 0][col] = 1 # C[0, 2b]
        W[(2*b+1) * n + 1][col] = 1 # C[1, 2b+1]
        
        # m4
        col = base_col + 3
        U[1 * m + 2][col] = 1   # A[1,2]
        V[2 * p + 2*b][col] = 1 # B[2, 2b]
        V[1 * p + 2*b][col] = -1 # B[1, 2b]
        W[2*b * n + 0][col] = 1 # C[0, 2b]
        W[2*b * n + 1][col] = 1 # C[1, 2b]
        
        # m5
        col = base_col + 4
        U[0 * m + 1][col] = 1   # A[0,1]
        U[0 * m + 2][col] = 1   # A[0,2]
        V[2 * p + 2*b+1][col] = 1 # B[2, 2b+1]
        W[2*b * n + 0][col] = -1 # C[0, 2b]
        W[2*b * n + 1][col] = 1 # C[0, 2b+1]
        
        # m6
        col = base_col + 5
        U[1 * m + 1][col] = 1   # A[1,1]
        U[0 * m + 1][col] = -1  # A[0,1]
        V[1 * p + 2*b][col] = 1 # B[1, 2b]
        V[1 * p + 2*b+1][col] = 1 # B[1, 2b+1]
        W[(2*b+1) * n + 1][col] = 1 # C[1, 2b+1]
        
        # m7
        col = base_col + 6
        U[0 * m + 2][col] = 1   # A[0,2]
        U[1 * m + 2][col] = -1  # A[1,2]
        V[2 * p + 2*b][col] = 1 # B[2, 2b]
        V[2 * p + 2*b+1][col] = 1 # B[2, 2b+1]
        W[2*b * n + 0][col] = 1 # C[0, 2b]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
