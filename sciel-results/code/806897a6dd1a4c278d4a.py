"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try to find a lower-rank decomposition using tensor decomposition
    # The multiplication tensor T has T[i,j,k] = 1 for the (i,j,k) triple
    # where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Build the tensor as a 3D array: T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # Reshape to matrix of shape (n*m, m*p) for each k, or use the standard form
    # T as (n*m) x (m*p) x (p*n) tensor where T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook rank is n*m*p = 105
    # Let's try to find if we can do better using SVD-based approaches
    
    # Construct the tensor in matrix form for each "slice"
    # For matrix multiplication, the tensor can be viewed as:
    # For each output index (i,k), the coefficient of A[i,j]*B[j,k] is 1
    
    # Let's try a random search for low-rank decompositions
    # using the fact that we need exact integer/half-integer solutions
    
    # First, let's verify the schoolbook solution works
    rank_school = n * m * p  # 105
    
    # Try to find a better solution using structured decomposition
    # Key insight: if we can group terms, we might reduce rank
    
    # For now, return the schoolbook solution which is guaranteed correct
    # But let's try one more thing: check if any symmetry helps
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # This is a fully structured tensor - no zeros, so hard to compress
    
    # Let's just return the schoolbook decomposition
    rank = rank_school
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
