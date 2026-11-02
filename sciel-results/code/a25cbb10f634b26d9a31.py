"""Program for matrix multiplication 3x5 x 5x7 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Strategy: Use block decomposition.
    - Split the 3x5 matrix into blocks: row blocks of sizes [1, 2], col blocks of sizes [1, 2, 2]
    - Split the 5x7 matrix into blocks: row blocks of sizes [1, 2, 2], col blocks of sizes [3, 4]
    - The 3x3 block (rows 1-3, cols 1-3 of A; rows 1-3, cols 1-3 of B) uses rank 23 (known optimal for 3x3x3)
    - Other blocks use schoolbook multiplication
    
    Actually, let's use a simpler and safer approach:
    - Use Strassen's algorithm on a 2x2x2 sub-block where applicable
    - The 3x5x5x7 problem can be decomposed into blocks
    
    Let me use a direct approach with known improvements:
    - For 3x3x3, use rank 23 decomposition (Landsberg)
    - For other blocks, use schoolbook
    
    Block structure:
    A is 3x5, B is 5x7
    Split A into row blocks [1, 2] and col blocks [1, 2, 2]
    Split B into row blocks [1, 2, 2] and col blocks [3, 4]
    
    This gives 2*3 = 6 row-block pairs for A, and 3*2 = 6 col-block pairs for B.
    The middle dimension (5) is split as [1, 2, 2] for both A's columns and B's rows.
    
    Blocks of A (i,j) where i in {0,1} (row block), j in {0,1,2} (col block):
    A[0,0]: 1x1, A[0,1]: 1x2, A[0,2]: 1x2
    A[1,0]: 2x1, A[1,1]: 2x2, A[1,2]: 2x2
    
    Blocks of B (j,k) where j in {0,1,2} (row block), k in {0,1} (col block):
    B[0,0]: 1x3, B[0,1]: 1x4
    B[1,0]: 2x3, B[1,1]: 2x4
    B[2,0]: 2x3, B[2,1]: 2x4
    
    Product C = A*B, split into row blocks [1,2] and col blocks [3,4]:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]  (1x3)
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]  (1x4)
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]  (2x3)
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]  (2x4)
    
    For each term A[i,j]*B[j,k], we need to multiply a (ri x mj) matrix by a (mj x pk) matrix.
    The rank for this block multiplication is ri * mj * pk (schoolbook) or less with Strassen.
    
    Let's compute the total rank with schoolbook:
    C[0,0]: 1*1*3 + 1*2*3 + 1*2*3 = 3 + 6 + 6 = 15
    C[0,1]: 1*1*4 + 1*2*4 + 1*2*4 = 4 + 8 + 8 = 20
    C[1,0]: 2*1*3 + 2*2*3 + 2*2*3 = 6 + 12 + 12 = 30
    C[1,1]: 2*1*4 + 2*2*4 + 2*2*4 = 8 + 16 + 16 = 40
    Total: 15 + 20 + 30 + 40 = 105 (same as schoolbook, as expected)
    
    To improve, we can use Strassen for 2x2 blocks:
    - A[1,1] is 2x2, B[1,1] is 2x3 (not square, so Strassen doesn't directly apply)
    - We need square blocks for Strassen
    
    Let me try a different block structure with square middle dimension blocks:
    Split middle dimension (5) into [2, 3]:
    A col blocks: [2, 3], B row blocks: [2, 3]
    
    A[0,0]: 3x2, A[0,1]: 3x3
    B[0,0]: 2x7, B[0,1]: 3x7
    
    C = A*B:
    C[:,0:7] = A[:,:2]*B[0:2,:] + A[:,2:5]*B[2:5,:]
    
    Term 1: 3x2 * 2x7, rank = 3*2*7 = 42
    Term 2: 3x3 * 3x7, rank = 3*3*7 = 63
    Total: 105
    
    No improvement. The issue is that Strassen requires square matrices.
    
    Let me just use the schoolbook decomposition which is guaranteed correct.
    """
    n, m, p = DIMENSIONS
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
