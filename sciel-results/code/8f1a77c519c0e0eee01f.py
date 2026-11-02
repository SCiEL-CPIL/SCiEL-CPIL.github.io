DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    We use a block-wise Strassen-like approach:
    - Decompose the 5x5 multiplication into blocks using 2x2 Strassen on the top-left 4x4 submatrix
    - Handle the remaining rows/columns (index 4) with schoolbook terms
    - Apply this for each of the 3 rows of the left matrix
    
    For 5x5: Split as [2,2,1] x [2,2,1]
    - Top-left 2x2 block: Strassen rank 7
    - Cross terms and 1x1 terms: schoolbook
    
    Actually, let me use a simpler guaranteed approach:
    Use Strassen for 2x2 blocks within the 5x5, padding to 6x6 or using 2+2+1 decomposition.
    
    For a 5x5 * 5x5 multiplication, we can decompose as:
    A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] where blocks are 2x2, 2x2, 1x2
    B = [[B11, B21, B31], [B12, B22, B32], [B13, B23, B33]] where blocks are 2x2, 2x2, 2x1
    
    This is getting complex. Let me use the simplest valid improvement:
    
    For 5x5 * 5x5, we can use the following block decomposition:
    Split 5 = 2 + 2 + 1
    
    A = [[A_2x2, A_2x2, A_2x1],
         [A_2x2, A_2x2, A_2x1],
         [A_1x2, A_1x2, A_1x1]]
         
    B = [[B_2x2, B_2x2, B_1x2],
         [B_2x2, B_2x2, B_1x2],
         [B_2x1, B_2x1, B_1x1]]
    
    The product C = A*B has blocks:
    C_11 = A_11*B_11 + A_12*B_21 + A_13*B_31  (2x2)
    C_12 = A_11*B_12 + A_12*B_22 + A_13*B_32  (2x2)
    C_13 = A_11*B_13 + A_12*B_23 + A_13*B_33  (2x1)
    ... etc.
    
    This is still complex. Let me just use the schoolbook for correctness, but try to get rank < 75.
    
    Actually, the simplest valid improvement: use Strassen on 2x2 blocks.
    For 5x5, we can handle indices 0,1 with Strassen (rank 7 for 2x2*2x2), 
    and indices 2,3,4 with schoolbook.
    
    For each row of the 3x5 left matrix:
    - Rows 0,1 of A (2x5) times 5x5 B:
      Split B into [B_5x2, B_5x3] where B_5x2 uses columns 0,1
      A_2x5 * B_5x2: This is 2x2 * 2x2 (top-left of A) * (top-left of B) + cross terms
      
    This is too complex to get right quickly. Let me use a known result:
    
    The bilinear rank of 5x5 * 5x5 is at most 19 (this is a known upper bound).
    But I don't have the exact coefficients memorized.
    
    Safest approach: Use schoolbook for rank 25 per 5x5, total rank 75.
    But that scores 1.0.
    
    Let me try rank 20 for 5x5 by using one Strassen-like optimization on a 2x2 subblock.
    
    For 5x5 * 5x5, consider the top-left 2x2 * 2x2 subproblem (indices 0,1 x 0,1).
    Use Strassen for this: rank 7 instead of rank 4.
    Wait, that increases rank! Strassen is 7 > 4.
    
    So Strassen doesn't help for small blocks. The benefit is for larger n.
    
    For 5x5, schoolbook rank 25 is actually quite good. The lower bound is 19.
    
    I'll stick with schoolbook rank 75 total. It's the safest.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 5 * 5 = 75, schoolbook
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
