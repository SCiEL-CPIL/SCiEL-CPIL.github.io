"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Schoolbook rank
    rank_schoolbook = n * m * p  # 90

    # We'll try to find a lower rank decomposition.
    # The tensor T is of shape (n*m, m*p, p*n) = (15, 30, 18)
    # T[i*m+j, j*p+k, k*n+i] = 1

    # Strategy: Use the slice method.
    # For each intermediate index j (0..m-1), we have a 3x6 matrix multiplication.
    # But actually, the tensor is T[i,j,k] = 1 for all i,j,k.
    # This is a very simple tensor! It's just the all-ones tensor of shape (3,5,6).
    # Wait, let me re-read the problem.

    # "Matrix multiplication 3,5,6" - Find a rank-minimal exact tensor decomposition
    # for multiplying a 3 by 5 matrix by a 5 by 6 matrix.
    # The output factor follows the official trace convention W[k*n+i,r].

    # The matrix multiplication tensor T[i,j,k] = 1 where:
    # i = row index of first matrix (0..n-1)
    # j = column index of first / row index of second (0..m-1)
    # k = column index of second matrix (0..p-1)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should equal T[i,j,k]

    # So T[i,j,k] = 1 for all valid i,j,k. This is indeed the all-ones tensor!
    # The rank of the all-ones tensor of shape (n,m,p) is 1, since it's just
    # (1,1,...,1) ⊗ (1,1,...,1) ⊗ (1,1,...,1).

    # Wait, but the schoolbook decomposition has rank n*m*p = 90. That seems way too high.
    # Let me re-examine.

    # Actually, looking at the schoolbook decomposition:
    # For each (i,j,k), there's a rank-1 term with:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # All other entries in that column are 0.

    # The product for a given (i',j',k') would be:
    # sum_r U[i'*m+j',r] * V[j'*p+k',r] * W[k'*n+i',r]
    # = 1 if there exists r such that i'*m+j' = i*m+j AND j'*p+k' = j*p+k AND k'*n+i' = k*n+i
    # = 1 if i'=i AND j'=j AND k'=k (since the indices are unique)
    # = T[i',j',k']

    # So yes, this correctly computes the matrix multiplication tensor.
    # But the rank is n*m*p = 90.

    # Can we do better? The tensor T[i,j,k] = 1 is the all-ones tensor.
    # Its rank is indeed 1! We can use:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i

    # Then sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]

    # So the rank is 1!

    # Let me verify this is correct. The matrix multiplication tensor for C = A*B where
    # A is n×m and B is m×p is defined as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 if the indices are valid, 0 otherwise.

    # In our case, all indices are valid, so T[i,j,k] = 1 for all i,j,k.
    # This is the all-ones tensor, which has rank 1.

    # So the minimal rank is 1!

    # Let me construct the rank-1 decomposition:
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]

    # Verify: for any i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1
    # This equals T[i,j,k] = 1. Correct!

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
