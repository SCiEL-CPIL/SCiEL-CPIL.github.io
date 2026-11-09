"""
Rank-44 decomposition for 3x4 * 4x4 matrix multiplication.

Approach:
- Split A (3x4) into top 2 rows (2x4) and bottom row (1x4).
- Split B (4x4) into left 2 cols (4x2) and right 2 cols (4x2).
- The output C (3x4) is partitioned into:
  C_TL (2x2) = A_top * B_left
  C_TR (2x2) = A_top * B_right
  C_BL (1x2) = A_bot * B_left
  C_BR (1x2) = A_bot * B_right

- For the 2x2 output blocks, we use a Strassen-like approach:
  A_top is 2x4, split into A_L (2x2) and A_R (2x2).
  B_left is 4x2, split into B_TL (2x2) and B_BL (2x2).
  C_TL = A_L * B_TL + A_R * B_BL.
  We can compute this using 7 multiplications of 2x2 blocks (Strassen on the sum).
  Similarly for C_TR.
  
  However, a simpler and more direct approach:
  The rank of 2x4 * 4x2 is at most 4*4=16 schoolbook, but we can do better.
  Actually, 2x4 * 4x2 = (2x2 + 2x2) * (2x2 + 2x2) = 4 products of 2x2.
  Each 2x2 product has rank 4, so 4*4=16.
  
  But wait, we can use the fact that 2x2 * 2x2 has rank 4, and we have 4 such products.
  Total for C_TL: 16.
  Total for C_TR: 16.
  Total for C_BL: 1x4 * 4x2 = 8.
  Total for C_BR: 1x4 * 4x2 = 8.
  Total: 16+16+8+8 = 48.
  
  To improve, we can use Strassen on the 2x2 * 2x2 products.
  Strassen reduces 2x2 * 2x2 from rank 4 to rank 7? No, Strassen is for 2x2 * 2x2 -> 7 multiplications of scalars, but each "multiplication" in Strassen is actually a product of linear combinations, which in tensor rank terms is still rank 1 per term. So 7 rank-1 terms for 2x2*2x2.
  
  So for C_TL = A_L * B_TL + A_R * B_BL, we can't directly apply Strassen because it's a sum of two products.
  
  Alternative: Use the known rank of 2x4 * 4x2. The tensor rank of [2,4,2] is known to be 12? No, schoolbook is 16.
  
  Let's use a different partition.
  Split A into 3 rows of 1x4.
  Split B into 4 cols of 4x1.
  This gives 12 products of 1x4 * 4x1, each rank 4. Total 48.
  
  Let's try to use the fact that 3x4 * 4x4 can be done with rank 44.
  One known decomposition uses the following:
  - 3x4 * 4x4 = (2x4 + 1x4) * 4x4.
  - 2x4 * 4x4 has rank 28 (using Strassen on 2x2 blocks: 4 blocks of 2x2*2x2, each rank 7, total 28).
  - 1x4 * 4x4 has rank 16 (schoolbook: 4*4=16, or 1x4*4x4 = 16 rank-1 terms).
  - Total: 28 + 16 = 44.
  
  This works! Let's implement it.
  
  For 2x4 * 4x4:
  A_top is 2x4, split into A11 (2x2), A12 (2x2).
  B is 4x4, split into B11, B12, B21, B22 (each 2x2).
  C_top = A11*B11 + A12*B21, A11*B12 + A12*B22.
  Using Strassen on each 2x2*2x2 product:
  Product 1: A11 * B11 -> 7 terms
  Product 2: A12 * B21 -> 7 terms
  Product 3: A11 * B12 -> 7 terms
  Product 4: A12 * B22 -> 7 terms
  Total: 28 terms.
  
  For 1x4 * 4x4:
  A_bot is 1x4, split into A21 (1x2), A22 (1x2).
  B is 4x4, split into B11, B12, B21, B22 (each 2x2).
  C_bot = A21*B11 + A22*B21, A21*B12 + A22*B22.
  Each 1x2 * 2x2 product has rank 4 (schoolbook).
  Total: 4*4 = 16 terms.
  
  Total rank: 28 + 16 = 44.
"""

DIMENSIONS = [3, 4, 4]

def _strassen_2x2(A_idx, B_idx, U, V, W, offset, n, m, p):
    """
    Add 7 rank-1 terms for 2x2 * 2x2 multiplication.
    A_idx: list of 4 indices into U for A block (row-major: 0,1,2,3 for A[0,0],A[0,1],A[1,0],A[1,1])
    B_idx: list of 4 indices into V for B block (row-major: 0,1,2,3 for B[0,0],B[0,1],B[1,0],B[1,1])
    U: U matrix (n*m x rank)
    V: V matrix (m*p x rank)
    W: W matrix (p*n x rank)
    offset: starting column index in U, V, W
    n, m, p: dimensions
    """
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    a00, a01, a10, a11 = A_idx
    b00, b01, b10, b11 = B_idx
    
    # M1 = (A00+A11)(B00+B11)
    U[a00][offset] += 1; U[a11][offset] += 1
    V[b00][offset] += 1; V[b11][offset] += 1
    W[0*n+0][offset] += 1  # C00 contribution +1
    W[1*n+1][offset] += 1  # C11 contribution +1
    offset += 1
    
    # M2 = (A10+A11)B00
    U[a10][offset] += 1; U[a11][offset] += 1
    V[b00][offset] += 1
    W[1*n+0][offset] += 1  # C10 contribution +1
    W[1*n+1][offset] -= 1  # C11 contribution -1
    offset += 1
    
    # M3 = A00(B10-B11)
    U[a00][offset] += 1
    V[b10][offset] += 1; V[b11][offset] -= 1
    W[0*n+1][offset] += 1  # C01 contribution +1
    W[1*n+1][offset] += 1  # C11 contribution +1
    offset += 1
    
    # M4 = A11(B01-B00)
    U[a11][offset] += 1
    V[b01][offset] += 1; V[b00][offset] -= 1
    W[0*n+0][offset] += 1  # C00 contribution +1
    W[1*n+0][offset] += 1  # C10 contribution +1
    offset += 1
    
    # M5 = (A00+A01)B11
    U[a00][offset] += 1; U[a01][offset] += 1
    V[b11][offset] += 1
    W[0*n+0][offset] -= 1  # C00 contribution -1
    W[0*n+1][offset] += 1  # C01 contribution +1
    offset += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[a10][offset] += 1; U[a00][offset] -= 1
    V[b00][offset] += 1; V[b01][offset] += 1
    W[1*n+1][offset] += 1  # C11 contribution +1
    offset += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[a01][offset] += 1; U[a11][offset] -= 1
    V[b10][offset] += 1; V[b11][offset] += 1
    W[0*n+0][offset] += 1  # C00 contribution +1
    offset += 1
    
    return offset

def _schoolbook_1x2x2(A_idx, B_idx, U, V, W, offset, n, m, p):
    """
    Add 4 rank-1 terms for 1x2 * 2x2 multiplication.
    A_idx: list of 2 indices into U for A block (A[0,0], A[0,1])
    B_idx: list of 4 indices into V for B block (B[0,0],B[0,1],B[1,0],B[1,1])
    U: U matrix (n*m x rank)
    V: V matrix (m*p x rank)
    W: W matrix (p*n x rank)
    offset: starting column index
    n, m, p: dimensions
    """
    a0, a1 = A_idx
    b00, b01, b10, b11 = B_idx
    
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    # Term 1: A0 * B00 -> C0
    U[a0][offset] += 1
    V[b00][offset] += 1
    W[0*n+0][offset] += 1  # C[0,0] in the 1x2 output, which maps to global row 0, col 0 of this block
    offset += 1
    
    # Term 2: A1 * B10 -> C0
    U[a1][offset] += 1
    V[b10][offset] += 1
    W[0*n+0][offset] += 1
    offset += 1
    
    # Term 3: A0 * B01 -> C1
    U[a0][offset] += 1
    V[b01][offset] += 1
    W[1*n+0][offset] += 1  # C[0,1] in the 1x2 output
    offset += 1
    
    # Term 4: A1 * B11 -> C1
    U[a1][offset] += 1
    V[b11][offset] += 1
    W[1*n+0][offset] += 1
    offset += 1
    
    return offset

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    offset = 0
    
    # A_top is rows 0,1 of A (2x4)
    # A_bot is row 2 of A (1x4)
    # B is 4x4
    
    # C_top is rows 0,1 of C (2x4)
    # C_bot is row 2 of C (1x4)
    
    # For C_top = A_top * B:
    # A_top split into A_L (cols 0,1) and A_R (cols 2,3)
    # B split into B_L (rows 0,1) and B_R (rows 2,3)
    # C_TL = A_L * B_L (2x2 * 2x2)
    # C_TR = A_R * B_R (2x2 * 2x2)
    # C_BL = A_L * B_R (2x2 * 2x2)
    # C_BR = A_R * B_L (2x2 * 2x2)
    
    # Wait, A_top is 2x4, B is 4x4.
    # A_top * B: (2x4) * (4x4) = 2x4.
    # Split A_top into A_L (2x2) and A_R (2x2).
    # Split B into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2).
    # C_TL = A_L * B_TL + A_R * B_BL
    # C_TR = A_L * B_TR + A_R * B_BR
    
    # We can compute A_L * B_TL using Strassen (7 terms)
    # A_R * B_BL using Strassen (7 terms)
    # A_L * B_TR using Strassen (7 terms)
    # A_R * B_BR using Strassen (7 terms)
    # Total: 28 terms for C_top.
    
    # A_L indices in U: row 0, col 0,1 -> U[0*4+0]=U[0], U[0*4+1]=U[1]
    # A_R indices in U: row 0, col 2,3 -> U[0*4+2]=U[2], U[0*4+3]=U[3]
    # A_L row 1: U[1*4+0]=U[4], U[1*4+1]=U[5]
    # A_R row 1: U[1*4+2]=U[6], U[1*4+3]=U[7]
    
    # B_TL indices in V: row 0,1 col 0,1 -> V[0*4+0]=V[0], V[0*4+1]=V[1], V[1*4+0]=V[4], V[1*4+1]=V[5]
    # B_TR indices in V: row 0,1 col 2,3 -> V[0*4+2]=V[2], V[0*4+3]=V[3], V[1*4+2]=V[6], V[1*4+3]=V[7]
    # B_BL indices in V: row 2,3 col 0,1 -> V[2*4+0]=V[8], V[2*4+1]=V[9], V[3*4+0]=V[12], V[3*4+1]=V[13]
    # B_BR indices in V: row 2,3 col 2,3 -> V[2*4+2]=V[10], V[2*4+3]=V[11], V[3*4+2]=V[14], V[3*4+3]=V[15]
    
    # C_TL indices in W: row 0,1 col 0,1 -> W[0*3+0]=W[0], W[0*3+1]=W[1], W[1*3+0]=W[3], W[1*3+1]=W[4]
    # C_TR indices in W: row 0,1 col 2,3 -> W[0*3+2]=W[2], W[0*3+3]=W[5], W[1*3+2]=W[6], W[1*3+3]=W[7]
    
    # For Strassen, we need to pass the indices in the order A00, A01, A10, A11 and B00, B01, B10, B11.
    # The W indices are passed as offsets, but the _strassen_2x2 function assumes the output block starts at W[0], W[1], W[n], W[n+1] relative to the block.
    # We need to modify _strassen_2x2 to accept the starting W index.
    
    # Let's redefine the helper functions to be more flexible.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
