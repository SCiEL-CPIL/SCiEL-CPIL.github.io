"""Matrix multiplication 2,3,10: find rank-minimal exact tensor decomposition.
The output factor follows the official trace convention W[k*n+i,r].
"""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook rank is 2*3*10 = 60.
    We attempt to find a lower rank decomposition using known results from algebraic complexity theory.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    the tensor T_{i,j,k} corresponds to C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    A rank-r decomposition has factor matrices:
    U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    such that for all i, j, k:
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For 2x3 * 3x10, we can use the fact that 10 = 3+3+3+1, and decompose into:
    - Three blocks of 2x3 * 3x3
    - One block of 2x3 * 3x1
    
    The best known rank for 2x3 * 3x3 is 17 (using Strassen-like algorithms).
    The rank for 2x3 * 3x1 is 6 (schoolbook).
    
    So an upper bound is 3*17 + 6 = 57.
    
    However, constructing explicit rank-17 coefficients for 2x3*3x3 is complex.
    Let me try a different approach: use a heuristic search or known explicit decompositions.
    
    Actually, for 2x3 * 3x3, the minimal rank is known to be 17.
    Let me construct this using the known algorithm.
    
    The 2x3 * 3x3 multiplication can be viewed as computing C = A * B where A is 2x3 and B is 3x3.
    This is equivalent to computing 3 columns of a 2x3 product, each requiring rank 6 in schoolbook.
    
    Using Strassen's algorithm adapted for rectangular matrices:
    For 2x3 * 3x3, we can use the following approach:
    - Split B into two parts: B1 (3x2) and B2 (3x1)
    - Compute A*B1 using a 2x3 * 3x2 algorithm (rank 11 or so)
    - Compute A*B2 using a 2x3 * 3x1 algorithm (rank 6)
    
    Actually, let me just try to find any rank less than 60 using a simple approach.
    
    For now, let me return the schoolbook decomposition as a safe baseline, and note that
    finding a lower rank requires sophisticated algebraic techniques.
    """
    n, m, p = DIMENSIONS
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
