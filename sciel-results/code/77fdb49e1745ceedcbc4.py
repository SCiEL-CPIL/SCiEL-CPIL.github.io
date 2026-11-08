"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x3 * 3x8 multiplication using Strassen for the 2x2 top-left block
    and schoolbook for the remaining terms.
    
    Matrix A (3x3) * B (3x8) = C (3x8)
    
    Partition:
      A = [[A11, A12, A13],     B = [[B11, B12, B13],
           [A21, A22, A23],          [B21, B22, B23],
           [A31, A32, A33]]          [B31, B32, B33]]
    
    where A11, A22 are 1x1 (scalars), A12, A21 are 1x2, A13, A23 are 1x3, etc.
    Actually let's use 2x2 blocks for Strassen:
    
    A = [[A11, A12, A13],     B = [[B11, B12, B13],
         [A21, A22, A23],          [B21, B22, B23],
         [A31, A32, A33]]          [B31, B32, B33]]
    
    For Strassen on 2x2 blocks (top-left 2x2 submatrices):
    A11 = A[0:2, 0:2] (2x2), A12 = A[0:2, 2:3] (2x1), A13 = A[0:2, 3:4]... 
    Wait, B is 3x8, so let me partition differently.
    
    Let me use: A is 3x3, B is 3x8.
    Split rows/cols of A into (2,1) and (1,2):
      A = [[A11, A12],       B = [[B11, B12],
           [A21, A22]]             [B21, B22]]
    where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
          B11 is 2x8, B12 is 2x0... no that doesn't work.
    
    Better: partition both into 2+1 rows/cols:
    A (3x3): rows split 2+1, cols split 2+1
      A11 = A[0:2,0:2] (2x2), A12 = A[0:2,2:3] (2x1)
      A21 = A[2:3,0:2] (1x2), A22 = A[2:3,2:3] (1x1)
    
    B (3x8): rows split 2+1, cols split 4+4
      B11 = B[0:2,0:4] (2x4), B12 = B[0:2,4:8] (2x4)
      B21 = B[2:3,0:4] (1x4), B22 = B[2:3,4:8] (1x4)
    
    C (3x8): rows split 2+1, cols split 4+4
      C11 = C[0:2,0:4] (2x4), C12 = C[0:2,4:8] (2x4)
      C21 = C[2:3,0:4] (1x4), C22 = C[2:3,4:8] (1x4)
    
    Using Strassen for the 2x2 block multiplication (top-left 2x2 of A times top-left 2x4 of B):
    M1 = (A11 + A22) * (B11 + B22)  -- but A22 is 1x1, B22 is 1x4, so this mixes sizes
    
    This is getting complicated. Let me just use the standard approach:
    Apply Strassen 2x2 to compute A[0:2,0:2] * B[0:2,0:8] and A[0:2,0:2] * B[0:2,4:8] etc.
    
    Actually, the cleanest approach: use the known rank-7 Strassen for 2x2*2x2,
    and handle the cross terms with schoolbook.
    
    Total terms:
    - A[0:2,0:2] * B[0:2,0:8]: 7 Strassen terms (2x2 * 2x8)
    - A[0:2,2:3] * B[2:3,0:8]: 2*1*8 = 16 schoolbook terms (2x1 * 1x8)
    - A[2:3,0:2] * B[0:2,0:8]: 1*2*8 = 16 schoolbook terms (1x2 * 2x8)
    - A[2:3,2:3] * B[2:3,0:8]: 1*1*8 = 8 schoolbook terms (1x1 * 1x8)
    
    Total = 7 + 16 + 16 + 8 = 47? That seems too low. Let me reconsider.
    
    Wait, for 2x2 * 2x8 Strassen:
    A11 (2x2) * B11 (2x8) where B11 = B[0:2, 0:8]
    Strassen: M1=(A11+A22)(B11+B22) but A22 doesn't exist in 2x2*2x8...
    
    I think the issue is that Strassen applies to square block matrices. Let me just do:
    - Use rank-7 Strassen for computing 2x2 * 2x2 subproblems
    - The 2x2 * 2x8 can be viewed as two 2x2 * 2x4 problems or handled directly
    
    Simplest correct approach: 
    Rank = 7 (for 2x2*2x2 part) + schoolbook for remaining
    = 7 + (2*1*8 + 1*2*8 + 1*1*8) = 7 + 16 + 16 + 8 = 47
    
    But wait, I need to be more careful. Let me just build it explicitly.
    """
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = 3, 3, 8  # A is n x m, B is m x p, C is n x p
    
    # We'll build the decomposition using:
    # 1. Strassen 2x2 for A[0:2,0:2] * B[0:2,0:8]  -> 7 terms
    # 2. Schoolbook for A[0:2,2:3] * B[2:3,0:8]    -> 2*1*8 = 16 terms  
    # 3. Schoolbook for A[2:3,0:2] * B[0:2,0:8]    -> 1*2*8 = 16 terms
    # 4. Schoolbook for A[2:3,2:3] * B[2:3,0:8]    -> 1*1*8 = 8 terms
    # Total rank = 7 + 16 + 16 + 8 = 47
    
    # Actually, let me reconsider the partition. For Strassen 2x2*2x2:
    # A_top = A[0:2, 0:2] (2x2), B_left = B[0:2, 0:8] (2x8)
    # But Strassen needs both to be 2x2. So I'll split B[0:2,0:8] into B[0:2,0:4] and B[0:2,4:8]
    # and apply Strassen to each: 7 + 7 = 14 terms for the 2x2 * 2x8 part.
    
    # Revised total:
    # 1. Strassen 2x2*2x4 for A[0:2,0:2]*B[0:2,0:4]: 7 terms
    # 2. Strassen 2x2*2x4 for A[0:2,0:2]*B[0:2,4:8]: 7 terms
    # 3. Schoolbook A[0:2,2:3]*B[2:3,0:8]: 2*1*8 = 16 terms
    # 4. Schoolbook A[2:3,0:2]*B[0:2,0:8]: 1*2*8 = 16 terms
    # 5. Schoolbook A[2:3,2:3]*B[2:3,0:8]: 1*1*8 = 8 terms
    # Total = 7 + 7 + 16 + 16 + 8 = 54
    
    # Hmm, still not optimal. Let me try a different partition.
    
    # Better: use 2x2 Strassen where possible.
    # A (3x3) = [[A11, A12, A13],  B (3x8) = [[B11, B12, B13, B14],
    #             [A21, A22, A23],               [B21, B22, B23, B24],
    #             [A31, A32, A33]]               [B31, B32, B33, B34]]
    # where each Bxj is 1x2 (so B is 3x8 = 3 rows, 4 cols of 1x2 blocks)
    
    # This is getting complex. Let me just use a simpler but still improved approach:
    # Use rank-7 Strassen for the 2x2*2x2 subproblem and schoolbook for the rest.
    
    # Partition: A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    #             B = [B11 B12; B21 B22] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    
    # C11 = A11*B11 + A12*B21  (2x4)
    # C12 = A11*B12 + A12*B22  (2x4)  
    # C21 = A21*B11 + A22*B21  (1x4)
    # C22 = A21*B12 + A22*B22  (1x4)
    
    # For A11*B11 (2x2 * 2x4): view as two 2x2*2x2 problems -> 7+7 = 14 terms
    # For A11*B12 (2x2 * 2x4): 7+7 = 14 terms
    # For A12*B21 (2x1 * 1x4): 2*1*4 = 8 terms
    # For A12*B22 (2x1 * 1x4): 8 terms
    # For A21*B11 (1x2 * 2x4): 1*2*4 = 8 terms
    # For A21*B12 (1x2 * 2x4): 8 terms
    # For A22*B21 (1x1 * 1x4): 4 terms
    # For A22*B22 (1x1 * 1x4): 4 terms
    # Total = 14+14+8+8+8+8+4+4 = 68. Not better than 72!
    
    # The issue is that splitting B into 4-column blocks doesn't help much.
    
    # Let me try: just use Strassen for 2x2*2x2 and keep the full 8 columns.
    # A11 (2x2) * B[0:2, 0:8] (2x8): This is 2x2 * 2x8.
    # I can apply Strassen-like: treat B[0:2,0:8] as [B11 B12] where B11=B[0:2,0:4], B12=B[0:2,4:8]
    # Then A11*B11 uses 7 terms, A11*B12 uses 7 terms. Total 14 for this part.
    
    # Full count:
    # A11*B[0:2,0:4]: 7
    # A11*B[0:2,4:8]: 7
    # A12*B[2:3,0:4]: 2*1*4 = 8
    # A12*B[2:3,4:8]: 8
    # A21*B[0:2,0:4]: 1*2*4 = 8
    # A21*B[0:2,4:8]: 8
    # A22*B[2:3,0:4]: 1*1*4 = 4
    # A22*B[2:3,4:8]: 4
    # Total = 7+7+8+8+8+8+4+4 = 54
    
    # That's 54, better than 72! Let me implement this.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 24 x 54
    W = [[0] * rank for _ in range(p * n)]  # 24 x 54
    
    col = 0
    
    # Strassen 2x2*2x4: A11 (2x2) * B11 (2x4)
    # A11 = A[0:2, 0:2], B11 = B[0:2, 0:4]
    # M1 = (A11[0]+A11[1]) * (B11[0]+B11[1])  -- but these are vectors
    # Actually for 2x2 * 2x4:
    # A11 = [[a00, a01], [a10, a11]], B11 = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C11 = A11 * B11 (2x4)
    # Strassen:
    # M1 = (a00+a11)*(b00+b10, b01+b11, b02+b12, b03+b13)
    # M2 = (a10+a11)*(b00, b01, b02, b03)
    # M3 = a00*(b10, b11, b12, b13)
    # M4 = a11*(b00, b01, b02, b03)
    # M5 = (a00+a01)*(b10+b11, b12+b13, ... ) -- this doesn't work cleanly for non-square
    
    # I think Strassen doesn't directly apply to 2x2*2x4 in the simple way.
    # Let me just use schoolbook for everything but try to find a better structure.
    
    # Actually, the known result: rank(3x3*3x8) is at most 54 using certain methods.
    # But let me just go with a clean schoolbook-based approach that's correct,
    # and if I can find a Strassen improvement, great.
    
    # For now, let me just return the schoolbook (rank 72) to ensure correctness,
    # but structure it so we can try improvements.
    
    # Let me try a different approach: use the fact that we can sometimes
    # share computations. But for exact decomposition with integer entries,
    # the schoolbook is hard to beat without careful algebraic structure.
    
    # Let me just do schoolbook for safety:
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
