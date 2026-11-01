"""Rewritten program: search for a low-rank exact tensor decomposition of 2x3 by 3x7 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# Known lower bound for 2x3 * 3x7 is 21 (from n*m*p / max(n,m,p) heuristic? Actually lower bound is max(n*m, m*p, p*n) = max(6,21,14)=21)
# Schoolbook rank is 42. We aim to find rank 21 or lower.

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = 1 if (i,j) in A, (j,k) in B, output (i,k).
    The tensor is over indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    For matmul C = A*B, C[i,k] = sum_j A[i,j] * B[j,k].
    The bilinear tensor T has T[i,j,k] = 1 for all i,j,k (since every product term contributes).
    Actually, the tensor decomposition is: C[i,k] = sum_r U[i, r] * V[k, r] ... no.
    
    Let me reconsider. The standard tensor for matrix multiplication of (n x m) and (m x p) is:
    T[i, j, k] where i indexes rows of A and C, j indexes columns of A / rows of B, k indexes columns of B and C.
    T[i, j, k] = 1 for all i, j, k.
    
    A rank-r decomposition gives: T[i,j,k] = sum_{r=1}^r U[i, r] * V[j, r] * W[k, r]
    where U is n x r, V is m x r, W is p x r.
    
    But the task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    And the output factor follows W[k*n+i, r].
    
    So the decomposition is:
    C[i, k] = sum_{j} A[i, j] * B[j, k] = sum_r (sum_j U[i*m + j, r] * A[i,j]) * (sum_k ... )
    
    Actually, the standard formulation:
    Let A be n x m, B be m x p.
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    A rank-r bilinear algorithm:
    C[i, k] = sum_{r=0}^{r-1} (sum_j U[i, j, r] * A[i, j]) * (sum_k V[j, k, r] * B[j, k])
    
    But that's not quite right either. Let me think about the standard tensor decomposition.
    
    The matrix multiplication tensor T ∈ R^{n x m x p} is defined by T[i, j, k] = 1 for all i, j, k.
    A rank-r decomposition is: T[i, j, k] = sum_{r=0}^{r-1} U[i, r] * V[j, r] * W[k, r]
    
    But the task specifies U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    This suggests a different grouping.
    
    Looking at the seed:
    U[i*m + j, column] = 1
    V[j*p + k, column] = 1
    W[k*n + i, column] = 1
    
    So the decomposition is:
    For each (i, j, k), we have a rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    The bilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * (something that picks out the right output)
    
    Actually, I think the convention is:
    The product C = A*B has entries C[i, k] = sum_j A[i, j] * B[j, k].
    
    The tensor T[i, j, k] = 1 for all i, j, k represents the fact that each A[i,j] and B[j,k] pair contributes to C[i,k].
    
    A rank-r decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that doesn't make sense dimensionally for a bilinear form. Let me re-read.
    
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    So U has rows indexed by (i, j), V by (j, k), W by (k, i).
    
    The decomposition condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    This is a 3-way tensor decomposition where the "tensor" T has entries T[(i,j), (j,k), (k,i)] = 1.
    
    So T is a 3-tensor with indices (i*m+j), (j*p+k), (k*n+i) and T = 1 for all valid (i,j,k).
    
    We need: sum_r U[a, r] * V[b, r] * W[c, r] = 1 where a = i*m+j, b = j*p+k, c = k*n+i.
    
    This is equivalent to the standard matrix multiplication tensor rank problem.
    """
    n, m, p = n, m, p
    # The tensor is implicitly 1 for all (i, j, k)
    # We don't need to build it explicitly, just verify the decomposition
    return n, m, p

def _verify(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _round_to_half_integer(M, tol=1e-8):
    """Round entries to nearest half-integer."""
    M2 = 2.0 * M
    M_rounded = np.round(M2)
    if np.max(np.abs(M2 - M_rounded)) > tol:
        return None
    return M_rounded / 2.0

def construct():
    n, m, p = DIMENSIONS
    
    # Fall back to schoolbook
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
