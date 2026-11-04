"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Approach: Use the known result that matrix multiplication of (n x m) * (m x p) 
    can be decomposed with rank less than n*m*p using bilinear algorithms.
    
    For 4x5 * 5x6, we use a combination of:
    1. Pad to 6x6 * 6x6 and use known optimal decompositions, OR
    2. Use a direct bilinear algorithm with fewer terms.
    
    Let's use the following approach:
    - We know that for square n x n matrix multiplication, the minimum rank is known for small n.
    - For 4x4 * 4x4, the minimum rank is 7 (Strassen) or possibly less with newer results.
    
    Actually, let me implement a valid decomposition using the following strategy:
    Split the problem into sub-problems and use Strassen where applicable.
    
    A (4x5), B (5x6), C (4x6)
    
    We can write:
    C = A * B
    
    Let's split A into A1 (4x2) and A2 (4x3), and B into B1 (2x6) and B2 (3x6).
    Then C = A1*B1 + A2*B2
    
    A1*B1: 4x2 * 2x6, rank = 4*2*6 = 48 (schoolbook)
    A2*B2: 4x3 * 3x6, rank = 4*3*6 = 72 (schoolbook)
    Total: 120, same as before.
    
    Better: Use the fact that we can share computations.
    
    Actually, let me just implement a valid bilinear algorithm that achieves rank < 120.
    
    Here's a concrete approach:
    For each column k of C, C[:,k] = A * B[:,k]
    This is a matrix-vector product for each k.
    
    But we need a bilinear algorithm that works for all inputs simultaneously.
    
    Let me use the following known result:
    The tensor rank of matrix multiplication M_{n,m,p} satisfies:
    R(M_{n,m,p}) <= min(R(M_{n,m,p'}), R(M_{n',m,p})) for suitable splits.
    
    For 4x5*5x6, let's try:
    R <= R(4x4*4x6) + R(4x1*1x6)
    
    R(4x1*1x6) = 4 (outer product)
    R(4x4*4x6): We can view this as 6 columns of 4x4*4x1, but they share A.
    Actually, 4x4*4x6 can be done with rank = 4*4*6/2 = 48? No, that's not right.
    
    Let me just implement a valid algorithm with rank 100 or so.
    
    Actually, the simplest improvement is to use the fact that we can combine terms.
    
    Let me implement a bilinear algorithm with the following structure:
    For i in 0..3, j in 0..4, k in 0..5:
      C[i,k] += A[i,j] * B[j,k]
    
    This is rank 120. To improve, we need to find linear dependencies.
    
    Given the complexity, let me just return a valid schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = n * m * p  # 120
    
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
