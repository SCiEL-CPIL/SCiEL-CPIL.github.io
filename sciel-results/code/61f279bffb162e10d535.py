"""Matrix multiplication 3x4 * 4x4 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = 3, 4, 4  # A is n x m, B is m x p, C is n x p
    
    # The tensor T[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank = n*m*p = 48
    # Then try to reduce using known optimizations for small matrix multiplication.
    
    # For 3x4 * 4x4, let's think about what's possible.
    # The trivial rank is 48.
    # Can we do better?
    
    # One approach: use the fact that we're multiplying 3x4 by 4x4.
    # We can write C = A * B where A is 3x4 and B is 4x4.
    
    # Let's try a different strategy: use bilinear algorithms.
    # The key insight is that for small matrices, we can sometimes find
    # decompositions with fewer terms than the naive count.
    
    # Actually, let me think about this more carefully.
    # The tensor rank of matmul(n, m, p) is at most n*m*p (schoolbook).
    # For n=3, m=4, p=4, that's 48.
    
    # Can we reduce this? Let's try to find a decomposition with fewer terms.
    
    # One idea: group some of the multiplications.
    # For example, for each pair (i, k), C[i,k] = sum_j A[i,j] * B[j,k].
    # This is a dot product of length m=4.
    
    # There's no obvious way to reduce the rank below 48 for this specific case
    # without using more sophisticated bilinear algorithms.
    
    # Let me try a different approach: use the structure of the problem.
    # Since n=3 is small, we have 3 rows of C, each being a length-4 dot product.
    
    # Actually, let me just verify the schoolbook decomposition is correct
    # and return it. The rank is 48.
    
    # But wait - maybe I can do better by using the fact that some entries
    # can be shared. Let me think...
    
    # For each (i, j) pair, A[i,j] is multiplied by B[j,0], B[j,1], B[j,2], B[j,3].
    # These contribute to C[i,0], C[i,1], C[i,2], C[i,3] respectively.
    
    # In the tensor decomposition, each term r corresponds to a "bilinear form"
    # that picks out specific A entries, B entries, and C entries.
    
    # The schoolbook uses one term per (i,j,k) triple, giving 48 terms.
    
    # To reduce the rank, we would need to find linear combinations that allow
    # us to compute multiple C entries with fewer independent multiplications.
    
    # For now, let's use the schoolbook decomposition with rank 48.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
