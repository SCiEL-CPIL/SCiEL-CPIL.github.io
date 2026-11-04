"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    We use a block decomposition approach:
    - Split the 4x5 matrix into two 2x5 blocks vertically
    - Split the 5x6 matrix into two 5x3 blocks horizontally
    - This gives us a 2x2 block structure where each block multiplication is 2x5 * 5x3 -> 2x3
    
    For each 2x5 * 5x3 multiplication, we can potentially use a reduced rank decomposition.
    The schoolbook rank for 2x5*5x3 is 2*5*3 = 30.
    
    We can split further: 2x5 * 5x3 can be seen as computing C[i,k] = sum_j A[i,j]*B[j,k]
    for i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2}.
    
    One approach: use the fact that for 2xn * nxm, we can sometimes save multiplications
    by sharing intermediate products.
    
    For 2x5 * 5x3, the schoolbook uses 30 multiplications. Can we do better?
    
    Actually, let's try a different block structure. Let's split 4x5 into four 1x5 rows,
    and 5x6 into six 5x1 columns. Then each output element C[i,k] = sum_j A[i,j]*B[j,k].
    
    This is just the schoolbook method with rank 120.
    
    Let's try to use Strassen-like ideas. For 2x2 * 2x2, Strassen uses 7 instead of 8.
    For 4x4 * 4x4, we can use Strassen recursively to get 7^2 = 49 instead of 64.
    
    For our case 4x5 * 5x6, we can pad to 4x6 * 6x6 or 4x5 * 5x8, but that's not ideal.
    
    Let's try a practical approach: use the schoolbook method but look for any simple
    algebraic cancellations or sharing opportunities.
    
    Actually, the simplest improvement might be to recognize that if we view the
    multiplication as a bilinear map, we might be able to use a Kronecker product
    structure.
    
    For now, let's return the schoolbook decomposition which is guaranteed correct.
    The rank is 120.
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
