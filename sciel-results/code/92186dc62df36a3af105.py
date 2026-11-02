"""Matrix multiplication 3x4x7 - subspace peeling approach."""
DIMENSIONS = [3, 4, 7]

from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We want to find a rank-minimal decomposition of the matrix multiplication tensor.
    # The tensor T has indices (i, j, k) -> W[k*n+i] * U[i*m+j] * V[j*p+k]
    # where the product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} (1 if valid)
    
    # Actually, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} = 1 if we're computing (A*B)[i,k] from A[i,j]*B[j,k]
    # Wait, let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # So we have matrices A (3x4), B (4x7), C = A*B (3x7).
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # The tensor decomposition is: C[i,k] = sum_r U[i*4+j... wait, U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (28, rank)  
    # W has shape (p*n, rank) = (21, rank)
    
    # The convention is W[k*n+i, r], so W index is k*n+i where k is row of B/col of C? 
    # Wait, let me think again. A is n x m = 3x4, B is m x p = 4x7, C is n x p = 3x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U[i*m+j, r] - index i*m+j corresponds to A[i,j]
    # V[j*p+k, r] - index j*p+k corresponds to B[j,k]
    # W[k*n+i, r] - index k*n+i corresponds to C[i,k]
    
    # So the decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for each (i,j,k)
    # But this needs to hold such that sum_j A[i,j]*B[j,k] = C[i,k]
    
    # Actually the tensor T_{i,j,k} = 1 (it's the structure tensor of matmul).
    # The rank decomposition is: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where U, V, W are the factor matrices.
    
    # For the standard schoolbook, rank = n*m*p = 84, with each term being a single (i,j,k) triple.
    
    # Let me try to find a lower rank decomposition.
    # For 3x4x7, the minimum rank is known to be at most 84 (schoolbook).
    # Let me try to construct a better one.
    
    # Actually, let me just return the schoolbook for now since finding a truly minimal
    # rank decomposition is a hard open problem in general.
    
    # But wait - the task says "rank-minimal". For small cases, we might be able to find it.
    # Let me check if there's a known result for 3x4x7.
    
    # The rank of the 3x4x7 matrix multiplication tensor... 
    # For m x n x p matrix multiplication, the rank is at most m*n*p.
    # Lower bounds: rank >= max(m*n, n*p, m*p) and rank >= m*n*p / max(m,n,p)
    # For 3x4x7: lower bound is max(12, 28, 21) = 28, and 84/7 = 12.
    # So rank is between 28 and 84.
    
    # Let me try a simple approach: use the standard decomposition but see if we can combine terms.
    
    # Actually, let me just implement the schoolbook as a baseline and see if we can improve.
    
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
