"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
from itertools import product

def _tensor_rank_via_mats_mult(n, m, p):
    """
    Compute the tensor rank of the matrix multiplication tensor T_{i,j,k}
    for multiplying an (n x m) matrix by an (m x p) matrix.
    
    The tensor T lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    We can compute its rank by flattening it into a matrix and doing rank decomposition.
    
    Approach: The tensor T can be viewed as a matrix of size (n*p) x (m*p) by fixing
    the middle index j. Specifically, T_{i,j,k} for fixed j gives a matrix M_j of size
    (n*p) x (n*p) ... no, that's not right.
    
    Actually, the standard way: T_{i,j,k} is a 3D array. To find its rank, we can
    flatten it as a matrix of size (n*m) x (m*p * p*n) or similar, but the rank of
    the flattened matrix doesn't directly give the tensor rank.
    
    The correct approach: The tensor rank is the minimum number of rank-1 tensors that
    sum to T. This is NP-hard in general, but for small tensors we can use the following:
    
    The tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... no.
    Actually T_{i,j,k} = 1 if (i,j,k) is a valid triple for matrix multiplication, i.e.,
    T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p). Wait, that's not right either.
    
    Let me reconsider. The matrix multiplication tensor is defined as:
    T_{i,j,k} = 1 if the indices correspond to a valid multiplication term.
    Specifically, for C[i,k] = sum_j A[i,j] B[j,k], the tensor is:
    T_{i,j,k} = 1 for all i, j, k (since each term A[i,j]*B[j,k] contributes to C[i,k]).
    
    Wait, that means T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    So T is just the all-ones tensor of size n x m x p? That can't be right because
    the rank would be 1.
    
    I think I'm confusing the tensor with the multiplication operation. Let me reconsider.
    
    The matrix multiplication tensor T_{i,j,k} is defined such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T_{i,j,k}
    
    And T_{i,j,k} = 1 if (i,j,k) is a valid index triple, which is all of them.
    So T_{i,j,k} = 1 for all i, j, k.
    
    But that would mean the tensor is rank 1, which is clearly wrong.
    
    I think the issue is that the tensor should be:
    T_{i,j,k} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... no.
    
    Let me look at this differently. The matrix multiplication map is:
    (A, B) -> A*B where A is n x m, B is m x p.
    
    The tensor T_{i,j,k} represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    So T_{i,j,k} = 1 for all i, j, k.
    
    But the rank of this tensor (as a 3D array) is not the same as the rank of the
    matrix multiplication operation. The rank of the matrix multiplication operation
    is the minimum number of bilinear forms needed to compute all entries of C.
    
    So we need: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Wait, that's not right. The correct form is:
    C[i,k] = sum_c U[i*m+?, c] * ... 
    
    Actually, the standard tensor decomposition for matrix multiplication is:
    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T_{i,j,k} = 1 for all i, j, k.
    
    So the tensor is the all-ones tensor of shape (n*m, m*p, p*n) ... no, the indices
    are (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    
    So T is an n x m x p tensor with all entries equal to 1.
    
    The rank of an all-ones tensor of shape (n, m, p) is min(n, m, p) if we're talking
    about the matrix rank of the flattened tensor, but the tensor rank is different.
    
    I think I need to just use the schoolbook construction and not try to compute the
    exact rank, since that's complex. Let me just use the schoolbook rank n*m*p = 36.
    """
    return n * m * p

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Use schoolbook construction
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
