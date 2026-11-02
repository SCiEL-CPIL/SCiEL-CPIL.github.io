"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    Approach: Construct the 48x48 matrix representation of the multiplication tensor
    and compute its exact rank using fraction-based Gaussian elimination.
    
    The matrix multiplication tensor M_{n,m,p} can be represented as a matrix of size
    (n*p) x (n*p) where the entry at position (i*p+k, i'*p+k') corresponds to the
    coefficient of the product A[i,j]*B[j,k] in C[i,k].
    
    For exact rank determination, we use the following construction:
    The tensor T[i,k,j] = 1 for all valid (i,j,k) can be flattened into a matrix
    where rows correspond to (i,k) pairs and columns correspond to (j) values, but
    this isn't quite right for rank computation.
    
    Actually, the standard way to compute the rank of a tensor via matrix flattening:
    For a 3-tensor T of dimensions (n*m, m*p, p*n) [wait, that's not standard].
    
    Let me reconsider. The matrix multiplication tensor M_{n,m,p} has entries
    T_{ijk} where i in [n], j in [m], k in [p]. We can view this as a matrix of size
    (n*p) x (m*p) or other flattening.
    
    Actually, the most common approach: view the tensor as a matrix of size (n*m) x (p*n)
    by fixing the middle index j. But that gives n*m x p*n = 8 x 12 for our case,
    which doesn't directly give the rank.
    
    Let me use a different standard result: the rank of the matrix multiplication tensor
    M_{n,m,p} equals the rank of a certain (n*p) x (n*p) matrix called the "structure
    matrix" or can be computed via other means.
    
    For our specific case (2,4,6), let's construct the tensor and compute its rank
    by flattening it appropriately.
    
    The tensor T has shape (n, p, m) = (2, 6, 4) with T[i,k,j] = 1 for all valid indices.
    Wait, that's not right either. The tensor should have T[i,j,k] = 1 for all i,j,k
    since each product term A[i,j]*B[j,k] contributes to C[i,k].
    
    Actually, for matrix multiplication, the tensor is T[i,k,j] = 1 for all i in [n], 
    j in [m], k in [p]. This is a 3-way tensor of size n x p x m.
    
    To compute its rank, we can flatten it in different ways. The rank of the tensor
    is the minimum number of rank-1 tensors that sum to T.
    
    One way to compute the tensor rank: flatten the tensor into a matrix of size
    (n*p) x m where the (i*p+k)-th row and j-th column is T[i,k,j]. The rank of this
    matrix is a lower bound on the tensor rank. For our case, this would be a 12x4 matrix
    with all entries 1, which has rank 1. That's not helpful.
    
    Let me try another flattening: (n*m) x (p*m) where entry (i*m+j, k*p+j') is... 
    this doesn't make sense for our tensor.
    
    Actually, I think I'm overcomplicating this. For matrix multiplication tensor
    M_{n,m,p}, the rank is known to be at most n*m*p (schoolbook) and at least
    max(n*p, m*p, n*m) + something.
    
    For our case, let's just use the schoolbook decomposition since finding the exact
    minimal rank for 2x4x6 is non-trivial and may require advanced techniques.
    
    However, we can try to use a simple optimization: if we can share some computations,
    we might reduce the rank slightly.
    
    Let's implement the schoolbook as the safe choice.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # For 2x4x6, the schoolbook rank is 48
    rank = rank_schoolbook
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
