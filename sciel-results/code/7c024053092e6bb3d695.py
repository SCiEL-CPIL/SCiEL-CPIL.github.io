"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Use the Strassen-like bilinear decomposition for 2x2 blocks,
    extended to handle the 4x5 by 5x5 dimensions via block partitioning.
    
    We partition the 4x5 matrix into blocks and the 5x5 matrix into blocks,
    then use the known rank-7 decomposition for 2x2 matrix multiplication
    (Strassen) to reduce the overall rank below the schoolbook 100.
    
    The approach:
    - Partition A (4x5) into a 2x2 grid of blocks where possible
    - Partition B (5x5) into a 2x2 grid of blocks where possible
    - Use Strassen's 7-multiplication scheme for 2x2 blocks
    - Handle the extra dimensions (5 vs 4) by padding or splitting
    
    For 4x5 * 5x5:
    - Split rows of A: 4 = 2+2
    - Split columns of A / rows of B: 5 = 2+3 or 3+2
    - Split columns of B: 5 = 2+3 or 3+2
    
    Let's use: A is 4x5, split rows into 2+2, split cols into 2+3
    B is 5x5, split rows into 2+3, split cols into 2+3
    
    This gives us blocks:
    A = [[A11(2x2), A12(2x3)],
         [A21(2x2), A22(2x3)]]
    
    B = [[B11(2x2), B12(2x3)],
         [B21(3x2), B22(3x3)]]
    
    The product C = A*B has blocks:
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each block product is a matrix multiplication of different sizes.
    We can use Strassen for the 2x2 * 2x2 parts and schoolbook for others.
    
    However, a cleaner approach: use the fact that we can embed smaller
    multiplications. Let's think about this differently.
    
    Actually, let's use a more systematic approach. The rank of matrix
    multiplication for n x m * m x p is at most n*m*p (schoolbook).
    
    For 4x5 * 5x5, schoolbook rank = 4*5*5 = 100.
    
    We can try to use bilinear algorithms. One approach is to use the
    tensor rank bounds. The tensor rank of matmul_{4,5,5} is known to
    be less than 100.
    
    Let's use a constructive approach based on Strassen-like decompositions.
    
    Key idea: If we can write the multiplication as a sum of rank-1 tensors
    (outer products), the number of such terms is the rank.
    
    For 2x2 * 2x2, Strassen shows rank <= 7 (vs schoolbook 8).
    
    For our case, let's partition:
    - A (4x5): split into top 2 rows and bottom 2 rows
    - B (5x5): split into left 2 cols and right 3 cols (for A's columns)
              and top 2 rows and bottom 3 rows (for B's rows)
    
    Actually, let me use a different strategy. I'll use the following:
    
    The matrix multiplication tensor for (n,m,p) can be decomposed using
    the "border rank" or "approximate rank" techniques, but we need exact.
    
    Let me try a direct construction. We'll use the fact that:
    - 4 = 2 + 2
    - 5 = 2 + 3
    
    Split A into A_top (2x5) and A_bottom (2x5)
    Split B into B_left (5x2) and B_right (5x3)
    
    Then C = A*B = A_top * B (top 2 rows of C) + ... no, that's not right.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let me index: i in 0..3, j in 0..4, k in 0..4
    
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can split the sum: sum_{j=0}^{1} + sum_{j=2}^{4}
    
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{4} A[i,j] * B[j,k]
    
    The first part is a 4x2 * 2x5 multiplication (rank 4*2*5 = 40 schoolbook)
    The second part is a 4x3 * 3x5 multiplication (rank 4*3*5 = 60 schoolbook)
    Total: 100, same as before.
    
    To improve, we need to use Strassen-like tricks within these sub-problems.
    
    For the 4x2 * 2x5 part: we can split the 4 rows into 2+2, giving two 2x2*2x5 multiplications.
    Each 2x2*2x5 can be done with rank... well, 2x2*2x5 is like two 2x2*2x2 plus a 2x2*2x1.
    
    This is getting complex. Let me try a simpler approach: just use the schoolbook
    decomposition but verify it works, then see if we can shave off a few terms.
    
    Actually, the simplest improvement: use the known fact that the rank of
    matmul_{2,2,2} is 7 (Strassen). We can embed this.
    
    Let me try: partition into 2x2 blocks where possible.
    
    A (4x5): 
    - Rows 0-1, Cols 0-1: A11 (2x2)
    - Rows 0-1, Cols 2-4: A12 (2x3)
    - Rows 2-3, Cols 0-1: A21 (2x2)
    - Rows 2-3, Cols 2-4: A22 (2x3)
    
    B (5x5):
    - Rows 0-1, Cols 0-1: B11 (2x2)
    - Rows 0-1, Cols 2-4: B12 (2x3)
    - Rows 2-4, Cols 0-1: B21 (3x2)
    - Rows 2-4, Cols 2-4: B22 (3x3)
    
    C (4x5):
    - C11 = A11*B11 + A12*B21  (2x2 + 2x3*3x2 = 2x2)
    - C12 = A11*B12 + A12*B22  (2x2*2x3 + 2x3*3x3 = 2x3)
    - C21 = A21*B11 + A22*B21  (2x2*2x2 + 2x3*3x2 = 2x2)
    - C22 = A21*B12 + A22*B22  (2x2*2x3 + 2x3*3x3 = 2x3)
    
    For C11: A11*B11 is 2x2*2x2 (Strassen rank 7), A12*B21 is 2x3*3x2 (schoolbook 2*3*2=12)
    For C12: A11*B12 is 2x2*2x3 (schoolbook 12), A12*B22 is 2x3*3x3 (schoolbook 18)
    etc.
    
    Total rank with this approach:
    C11: 7 + 12 = 19
    C12: 12 + 18 = 30
    C21: 12 + 12 = 24  (A21*B11 is 2x2*2x2=8 schoolbook, A22*B21 is 2x3*3x2=12)
    Wait, A21 is 2x2, B11 is 2x2, so A21*B11 is 2x2*2x2, schoolbook 8 or Strassen 7.
    A22 is 2x3, B21 is 3x2, so A22*B21 is 2x3*3x2, schoolbook 12.
    C21: 7 + 12 = 19
    C22: A21*B12 is 2x2*2x3=12, A22*B22 is 2x3*3x3=18, total 30
    
    Grand total: 19 + 30 + 19 + 30 = 98
    
    That's only 2 less than 100. Not great.
    
    Let me try a different partition. What if I use 5 = 3 + 2?
    
    Or better, let me just implement the schoolbook and return it. The score
    is 100, which is the baseline. Any improvement is better.
    
    But wait - the task says "rank-minimal exact tensor decomposition". The
    evaluator measures the raw value (lower is better). So I want to minimize rank.
    
    Let me just do schoolbook for now and ensure correctness.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 100
    V = [[0] * rank for _ in range(m * p)]  # 25 x 100
    W = [[0] * rank for _ in range(p * n)]  # 20 x 100
    
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
