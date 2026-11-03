"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    
    Strategy:
    - Start with schoolbook decomposition (rank 50)
    - Use Strassen-style block decomposition where possible
    - For 2x2 sub-blocks, Strassen reduces rank from 8 to 7
    - Combine blocks to minimize total rank
    
    The matrix multiplication tensor for C = A*B where A is 2x5 and B is 5x5:
    - A has shape (2, 5)
    - B has shape (5, 5)  
    - C has shape (2, 5)
    
    We can view this as multiplying (2x5) * (5x5) = 2x5
    
    One approach: split the 5 columns of A and 5 rows of B into blocks.
    Since 5 = 2 + 2 + 1, we can use a block structure.
    
    However, a simpler approach: note that we can use bilinear complexity results.
    For small matrices, the known bounds are:
    - M(2,5,5): The rank is at most 25 (trivial) but can be lower.
    
    Actually, let's think about this more carefully. The tensor rank for matrix multiplication
    M(n,m,p) is the bilinear complexity. For M(2,5,5):
    
    We can decompose this using the fact that:
    M(2,5,5) <= M(2,2,2) + M(2,2,2) + M(2,1,5) + M(2,1,5) + M(2,1,1)
    But this isn't quite right.
    
    Let me use a different approach. We can write the multiplication as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For each (i,k) pair, we have a linear combination of products A[i,j]*B[j,k].
    
    Key insight: We can use the Strassen algorithm structure. For 2x2x2, Strassen uses
    7 multiplications instead of 8. 
    
    Let's decompose the 5x5 B matrix and 5 columns of A into blocks:
    - Split into 2+2+1 for both dimensions
    
    Actually, let me try a more direct approach. The rank of M(2,5,5) is known to be 
    at most 25 (schoolbook). But we can do better.
    
    One known result: M(2,5,5) <= 25, and it's conjectured to be around 20-22.
    
    Let me implement a search-based approach:
    1. Start with schoolbook (rank 50)
    2. Try to find linear dependencies among the terms
    3. Use randomized linear algebra to find the rank
    
    Actually, for the purposes of this task, let me implement a constructive approach
    that achieves rank less than 50.
    
    Key idea: Use the fact that for matrix multiplication, we can sometimes share
    intermediate products. For example, in Strassen's algorithm:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    This reduces 8 multiplications to 7.
    
    For our case, let's split the 5 columns into 2+2+1 and 5 rows into 2+2+1:
    
    A = [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    B = [B1; B2; B3] where B1 is 2x5, B2 is 2x5, B3 is 1x5
    
    Then C = A1*B1 + A2*B2 + A3*B3
    
    Each term:
    - A1*B1: 2x2 * 2x5 = 2x5, rank at most 2*2*5=20, but can be lower
    - A2*B2: 2x2 * 2x5 = 2x5, rank at most 20
    - A3*B3: 2x1 * 1x5 = 2x5, rank at most 2*1*5=10
    
    Total: at most 20 + 20 + 10 = 50 (same as schoolbook)
    
    But we can do better! For 2x2 * 2x5, we can use Strassen-like techniques.
    
    Actually, let me just implement a known good decomposition. The rank of M(2,5,5)
    is known to be at most 25 (by various results). Let me aim for rank 25.
    
    Wait, that's the same as schoolbook for 5x5. Let me reconsider.
    
    Schoolbook for 2x5 * 5x5: rank = 2*5*5 = 50.
    
    We can improve this. One approach: use the fact that M(2,5,5) <= M(2,3,3) + M(2,2,2) + M(2,2,2) + M(2,1,2) + M(2,1,2) + ... 
    This is getting complicated.
    
    Let me just implement a simple improvement: use Strassen for the 2x2 sub-blocks.
    
    Split A into [A1 A2 A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    Split B into [B1; B2; B3] where B1 is 2x5, B2 is 2x5, B3 is 1x5
    
    For A1*B1 (2x2 * 2x5): We can think of this as 5 separate 2x2 * 2x1 problems.
    For each column of B1, we have a 2x2 * 2x1 multiplication.
    Using Strassen-like: M(2,2,1) has rank at most 4 (schoolbook) but can it be lower?
    Actually M(2,2,1) is just computing 2 dot products of length 2, so rank 4 is tight.
    
    Hmm, let me try a different approach. Let me just return the schoolbook solution
    but verify it's correct, and note that finding a better one requires more sophisticated
    tensor decomposition techniques.
    
    For now, I'll implement the schoolbook solution with rank 50.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = n * m * p  # 50
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 10 x 50
    V = [[0] * rank for _ in range(m * p)]  # 25 x 50
    W = [[0] * rank for _ in range(p * n)]  # 10 x 50
    
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
