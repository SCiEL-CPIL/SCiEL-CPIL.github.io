"""Matrix multiplication 2,2,16: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 16]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Return {'U','V','W'} factor matrices for matrix multiplication 2x2 by 2x16.
    U: (n*m, rank) = (4, rank)
    V: (m*p, rank) = (32, rank)
    W: (p*n, rank) = (32, rank)
    Trace convention: W[k*n + i, r]
    
    Schoolbook gives rank = 64. We try to find a lower rank via
    structured search exploiting the symmetry of the problem.
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 64
    
    # For 2x2 * 2x16, we can use the fact that the 2x2 multiplication
    # can be done in rank 7 (Strassen-like) but for exact integer/half-integer
    # we need to be careful. Actually, for 2x2 matrix mult, the known exact
    # rank over Q is 7 (Strassen). But wait - Strassen uses 7 multiplications
    # for 2x2*2x2. However, the tensor rank of 2x2 matrix multiplication is 7.
    
    # For 2x2 * 2x16:
    # C = A * B where A is 2x2, B is 2x16, C is 2x16
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0,...,15}
    # T[i,j,k] = 1 for all (i,j,k) -- this is the "all-ones" tensor of shape (2,2,16)
    # Wait, no. The matrix multiplication tensor for A(2x2) * B(2x16) = C(2x16) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor has T[i, j, k] = 1 for all i in [0,1], j in [0,1], k in [0,15]
    # This is simply the outer product of ones vectors: T = 1_2 ⊗ 1_2 ⊗ 1_16
    # So the tensor rank is 1! Because it's a rank-1 tensor.
    
    # Wait, that's the standard matrix multiplication tensor. Let me reconsider.
    # The tensor for matrix multiplication (n x m) * (m x p) = (n x p) is:
    # T[i, j, k] where i indexes rows of A (and C), j indexes cols of A (and rows of B), 
    # k indexes cols of B (and C).
    # T[i,j,k] = 1 for all valid i,j,k.
    # This IS a rank-1 tensor: T[i,j,k] = 1*1*1.
    
    # So the rank is 1! U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r 1 * 1 * 1 = rank
    # But we need this to equal T[i,j,k] = 1.
    # So if rank = 1, sum = 1. Correct!
    
    # Wait, but that seems too easy. Let me re-read the problem.
    # "multiplying a 2 by 2 matrix by a 2 by 16 matrix"
    # A is 2x2, B is 2x16, C = A*B is 2x16
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    # The tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Hmm, this is the standard formulation where the tensor T has entries T[i,j,k]=1.
    # And indeed T = 1 ⊗ 1 ⊗ 1 is rank 1.
    
    # So the answer is rank 1!
    rank = 1
    U = [[1] * rank for _ in range(n * m)]  # (4, 1)
    V = [[1] * rank for _ in range(m * p)]  # (32, 1)
    W = [[1] * rank for _ in range(p * n)]  # (32, 1)
    
    # Verify: for all i in [0,1], j in [0,1], k in [0,15]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # = 1 * 1 * 1 = 1 = T[i,j,k] ✓
    
    return {"U": U, "V": V, "W": W}
