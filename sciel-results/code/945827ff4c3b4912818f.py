"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W):
    """
    Add a Strassen 2x2x2 block at given offsets.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+3 (j=0,1; k=0,1)  
    W rows: offset_w to offset_w+3 (k=0,1; i=0,1)
    Returns number of columns used (7).
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][offset_u + col] = 1  # A00
    U[offset_u + 3][offset_u + col] = 1  # A11
    V[offset_v + 0][offset_u + col] = 1  # B00
    V[offset_v + 3][offset_u + col] = 1  # B11
    W[offset_w + 0][offset_u + col] = 1  # C00 += M1
    W[offset_w + 3][offset_u + col] = 1  # C11 += M1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][offset_u + col] = 1  # A10
    U[offset_u + 3][offset_u + col] = 1  # A11
    V[offset_v + 0][offset_u + col] = 1  # B00
    W[offset_w + 2][offset_u + col] = 1  # C10 += M2
    W[offset_w + 3][offset_u + col] = -1 # C11 -= M2
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][offset_u + col] = 1  # A00
    V[offset_v + 1][offset_u + col] = 1  # B01
    V[offset_v + 3][offset_u + col] = -1 # B11
    W[offset_w + 0][offset_u + col] = 1  # C00 += M3
    W[offset_w + 3][offset_u + col] = 1  # C11 += M3
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][offset_u + col] = 1  # A11
    V[offset_v + 2][offset_u + col] = 1  # B10
    V[offset_v + 0][offset_u + col] = -1 # B00
    W[offset_w + 0][offset_u + col] = 1  # C00 += M4
    W[offset_w + 2][offset_u + col] = 1  # C10 += M4
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][offset_u + col] = 1  # A00
    U[offset_u + 1][offset_u + col] = 1  # A01
    V[offset_v + 3][offset_u + col] = 1  # B11
    W[offset_w + 0][offset_u + col] = -1 # C00 -= M5
    W[offset_w + 1][offset_u + col] = 1  # C01 += M5
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][offset_u + col] = 1  # A10
    U[offset_u + 0][offset_u + col] = -1 # A00
    V[offset_v + 0][offset_u + col] = 1  # B00
    V[offset_v + 1][offset_u + col] = 1  # B01
    W[offset_w + 3][offset_u + col] = 1  # C11 += M6
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][offset_u + col] = 1  # A01
    U[offset_u + 3][offset_u + col] = -1 # A11
    V[offset_v + 2][offset_u + col] = 1  # B10
    V[offset_v + 3][offset_u + col] = 1  # B11
    W[offset_w + 0][offset_u + col] = 1  # C00 += M7
    col += 1
    
    return col

def _schoolbook_2x2x1(offset_u, offset_v, offset_w, U, V, W):
    """
    Add a schoolbook 2x2x1 block at given offsets.
    U rows: offset_u to offset_u+3 (i=0,1; j=0,1)
    V rows: offset_v to offset_v+2 (j=0,1; k=0)
    W rows: offset_w to offset_w+2 (k=0; i=0,1)
    Returns number of columns used (4).
    """
    col = 0
    for i in range(2):
        for j in range(2):
            U[offset_u + i*2 + j][offset_u + col] = 1
            V[offset_v + j][offset_u + col] = 1
            W[offset_w + i][offset_u + col] = 1
            col += 1
    return col

def _construct_hybrid():
    """
    Construct a hybrid Strassen+schoolbook decomposition for 2x4x5.
    
    Strategy:
    - Split the 4 columns of A into two 2-column blocks: A = [A_left, A_right]
    - Split the 5 rows of B into: B = [B_top (2 rows), B_bottom (3 rows)]
    - For each (A_block, B_block) pair, use appropriate decomposition
    
    Actually, let's think more carefully:
    C = A * B where A is 2x4, B is 4x5
    
    We can split A into two 2x2 blocks: A = [A1, A2] where A1 is cols 0-1, A2 is cols 2-3
    We can split B into blocks: B = [B1; B2] where B1 is rows 0-1 (2x5), B2 is rows 2-4 (3x5)
    
    Then C = A1*B1 + A2*B2
    
    For A1*B1: 2x2 * 2x5 = 2x5
    Split B1 into [B1a, B1b] where B1a is 2x2 (cols 0-1), B1b is 2x3 (cols 2-4)
    A1*B1a: 2x2 * 2x2 -> use Strassen (rank 7)
    A1*B1b: 2x2 * 2x3 -> need decomposition
    
    For A2*B2: 2x2 * 3x5
    Split B2 into [B2a; B2b] where B2a is 2x5 (rows 0-1 of B2), B2b is 1x5 (row 2 of B2)
    A2*B2a: 2x2 * 2x5 -> similar to above
    A2*B2b: 2x2 * 1x5 -> schoolbook (rank 4)
    
    Let me use a simpler approach:
    Split into 4 blocks of 2x2 for A and 5 blocks of 1x1 for B? No, that's schoolbook.
    
    Better: Use the known result that 2x4x5 can be done with rank 36 by:
    - Two 2x2x2 Strassen blocks (rank 7 each) = 14
    - Plus schoolbook for remaining terms
    
    Let me try a different decomposition:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into groups: j in {0,1} and j in {2,3}
    C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    First term: 2x2 * 2x5
    Second term: 2x2 * 2x5
    
    For each 2x2 * 2x5, split the 5 columns into 2+2+1:
    2x2 * 2x2: Strassen (rank 7)
    2x2 * 2x1: schoolbook (rank 4)  
    2x2 * 1x2: schoolbook (rank 4)
    
    Total per 2x2*2x5: 7 + 4 + 4 = 15
    Total for both: 30
    
    Wait, let me reconsider. 2x2 * 2x5:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    ...
    
    For the 2x2*2x2 part (cols 0-1), use Strassen: rank 7
    For col 2: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2], C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    This is 2x2*1x2, schoolbook rank 4
    Similarly for cols 3,4: another 2x2*1x2 each, rank 4 each
    
    So per 2x2*2x5: 7 + 4 + 4 + 4 = 19? That seems high.
    
    Actually for 2x2*1x2 (one column of B), we have:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    This needs 4 scalar products, so rank 4.
    
    Total: 2 * (7 + 4 + 4 + 4) = 2 * 19 = 38. Worse than schoolbook 40? No, 38 < 40.
    
    But the cell finding says 36. Let me try:
    2x2*2x5 = 2x2*2x2 (Strassen 7) + 2x2*2x3
    2x2*2x3: split as 2x2*2x2 (Strassen 7) + 2x2*2x1 (schoolbook 4)? No, 2x2*2x3 is 2x2 times a 2x3 matrix.
    
    Hmm, let me just implement the 36-rank construction from the cell finding:
    "two 2x2x5 blocks, each further decomposed into two 2x2x2 Strassen blocks (rank 7) and one 2x2x1 schoolbook block (rank 4)"
    
    Wait, 2x2x5 doesn't make sense as a tensor. Let me re-read:
    "Decomposing the 2x4x5 multiplication into two 2x2x5 blocks"
    
    I think they mean: split A's 4 columns into two groups of 2, giving two subproblems of type 2x2x5.
    Each 2x2x5 is further decomposed into:
    - One 2x2x2 block (Strassen, rank 7)  
    - One 2x2x1 block... but 2x2x1 would be 2x2 * 1x5? That doesn't work dimensionally.
    
    Let me just try to construct rank 36 directly.
    """
    n, m, p = 2, 4, 5
    # Total rank: 2 * (7 + 4 + 4) = 30? Or some other combination?
    # Let me try: split into 2 blocks of 2x2x5
    # Each 2x2x5: use Strassen for first 2 cols (rank 7), schoolbook for next 3 cols (rank 6? no, 2x2*3 = 6 products)
    # 2x2*2x3: C[i,k] for k=0,1,2. That's 2*3=6 entries, each is sum of 2 products.
    # Can we do better than schoolbook 6? For 2x2*2x3, the tensor has rank... 
    # Actually 2x2*2x3 is just 6 independent scalar products if we view it as 2x2 times 2x3.
    # No wait, it's a bilinear map. The tensor M2x2x2x3 has rank at most 6 (schoolbook).
    # Can it be done in less? Probably not for exact.
    
    # So: 2 * (7 + 6) = 26? That seems too optimistic.
    
    # Let me just go with a safe construction and verify it.
    
    # Simple approach: use Strassen for 2x2x2 subproblems where possible
    # A is 2x4, B is 4x5
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x2
    # Split B = [B1; B2] where B1 is 2x5, B2 is 2x5
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x5):
    #   Split B1 = [B1a | B1b | B1c] where B1a is 2x2, B1b is 2x1, B1c is 2x2
    #   A1*B1a: 2x2*2x2 -> Strassen rank 7
    #   A1*B1b: 2x2*2x1 -> schoolbook rank 4
    #   A1*B1c: 2x2*2x2 -> Strassen rank 7
    #   Subtotal: 18
    
    # For A2*B2 (2x2 * 2x5): same as above, subtotal 18
    # Total: 36
    
    # Let's implement this!
    
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Block 1: A1 (rows 0-1, cols 0-1 of A) * B1 (rows 0-1 of B, all 5 cols)
    # A1 is at U rows 0-3 (i=0,1; j=0,1)
    # B1 is at V rows 0-9 (j=0,1; k=0..4)
    
    # Sub-block 1a: A1 * B1a (B1a is rows 0-1, cols 0-1)
    # Strassen 2x2x2
    col_offset += _strassen_2x2x2(0, 0, 0, U, V, W)  # U offset 0, V offset 0, W offset 0
    
    # Sub-block 1b: A1 * B1b (B1b is rows 0-1, col 2)
    # Schoolbook 2x2x1 (one column of B)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U[0][col_offset] = 1; V[2][col_offset] = 1; W[4][col_offset] = 1  # A00*B02 -> C0,2
    col_offset += 1
    U[1][col_offset] = 1; V[3][col_offset] = 1; W[4][col_offset] = 1  # A01*B12 -> C0,2
    col_offset += 1
    U[2][col_offset] = 1; V[2][col_offset] = 1; W[5][col_offset] = 1  # A10*B02 -> C1,2
    col_offset += 1
    U[3][col_offset] = 1; V[3][col_offset] = 1; W[5][col_offset] = 1  # A11*B12 -> C1,2
    col_offset += 1
    
    # Sub-block 1c: A1 * B1c (B1c is rows 0-1, cols 3-4)
    # Strassen 2x2x2 for cols 3-4
    # U offset 0, V offset 6 (j=0,1; k=3,4), W offset 6 (k=3,4; i=0,1)
    col_offset += _strassen_2x2x2(0, 6, 6, U, V, W)
    
    # Block 2: A2 (rows 0-1, cols 2-3 of A) * B2 (rows 2-3 of B, all 5 cols)
    # A2 is at U rows 4-7 (i=0,1; j=2,3)
    # B2 is at V rows 10-19 (j=2,3; k=0..4)
    
    # Sub-block 2a: A2 * B2a (B2a is rows 2-3, cols 0-1)
    # Strassen 2x2x2
    # U offset 4, V offset 10, W offset 0
    col_offset += _strassen_2x2x2(4, 10, 0, U, V, W)
    
    # Sub-block 2b: A2 * B2b (B2b is rows 2-3, col 2)
    # Schoolbook
    # C[0,2] += A[0,2]*B[2,2] + A[0,3]*B[3,2]
    # C[1,2] += A[1,2]*B[2,2] + A[1,3]*B[3,2]
    U[4][col_offset] = 1; V[12][col_offset] = 1; W[4][col_offset] = 1  # A02*B22 -> C0,2
    col_offset += 1
    U[5][col_offset] = 1; V[13][col_offset] = 1; W[4][col_offset] = 1  # A03*B32 -> C0,2
    col_offset += 1
    U[6][col_offset] = 1; V[12][col_offset] = 1; W[5][col_offset] = 1  # A12*B22 -> C1,2
    col_offset += 1
    U[7][col_offset] = 1; V[13][col_offset] = 1; W[5][col_offset] = 1  # A13*B32 -> C1,2
    col_offset += 1
    
    # Sub-block 2c: A2 * B2c (B2c is rows 2-3, cols 3-4)
    # Strassen 2x2x2
    # U offset 4, V offset 16 (j=2,3; k=3,4), W offset 6 (k=3,4; i=0,1)
    col_offset += _strassen_2x2x2(4, 16, 6, U, V, W)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Use the constructive hybrid decomposition
    result = _construct_hybrid()
    
    # Verify
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify(U, V, W, n, m, p, rank):
        return result
    
    # Fallback to schoolbook if verification fails
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
