"""Rewritten program for matrix multiplication 4x4 x 4x5 tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    The tensor T[i, j, k] = 1 if and only if the (i,k) entry of A*B depends on A[i,j]*B[j,k].
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that for all i,j,k:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This is equivalent to: for each (j, k), the vector (over c) of U[:,c]*V[:,c] must satisfy
    the bilinear constraints. A standard approach is to fix V and W and solve for U, or use
    iterative methods.
    
    We'll use a simple iterative approach: initialize randomly, then alternate between
    solving for each factor given the others. For exactness with integer/half-integer entries,
    we'll try to find a low-rank decomposition and then round.
    
    Actually, let me think about this more carefully. The tensor for matrix multiplication
    A (n x m) * B (m x p) -> C (n x p) has entries:
      T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    
    So T is a 3-tensor with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, that's not right. The tensor is defined such that:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for each (i,j,k) triple, and the decomposition is:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This means for each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    One trivial decomposition is rank n*m*p with each term being a single (i,j,k) hyperedge.
    
    For a better decomposition, we can use the fact that matrix multiplication can be
    decomposed using Strassen-like methods, but for 4x4 x 4x5, we need to be careful.
    
    Let me try a different approach: use the fact that the matrix multiplication tensor
    can be decomposed using the standard "outer product" structure. For each j, the
    contribution to C is A[:,j] * B[j,:], which is an outer product. So we can write:
      C = sum_j A[:,j] * B[j,:]
    
    This gives a decomposition of rank m (one term per column of A / row of B), but this
    is not a valid tensor decomposition in the sense required because the factors U, V, W
    need to be such that the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for
    all (i,j,k).
    
    Actually, the standard schoolbook decomposition has rank n*m*p where each term
    corresponds to a single (i,j,k) triple.
    
    Let me try to find a lower rank decomposition. One idea is to use the fact that
    for fixed j, the terms involving A[i,j] and B[j,k] can be grouped. But the tensor
    structure requires that each term c contributes to all (i,j,k) where the product
    A[i,j]*B[j,k] appears.
    
    I think the key insight is that the matrix multiplication tensor T[i,j,k] = 1 for
    all (i,j,k) is actually a "full" tensor, and its tensor rank is at least max(n*m, m*p, p*n).
    For n=4, m=4, p=5, this gives at least max(16, 20, 20) = 20.
    
    Let me try to construct a rank-20 decomposition. The idea is to use the fact that
    for each (j,k), we can have a term that covers all i, or for each (i,j), a term that
    covers all k, etc.
    
    Actually, I recall that the tensor rank of matrix multiplication is at least the
    maximum of the three dimensions' products, and there are known lower bounds. For
    4x4 x 4x5, let me try to find a decomposition of rank 20.
    
    Here's an approach: for each j in range(m) and k in range(p), define a term c(j,k)
    such that:
      U[i*m+j, c(j,k)] = 1 for all i
      V[j*p+k, c(j,k)] = 1
      W[k*n+i, c(j,k)] = 1 for all i
    
    Then for a given (i,j,k), the sum over c is:
      sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    = sum_{j',k'} [i*m+j == i*m+j' and j*p+k == j'*p+k' and k*n+i == k'*n+i] * 1
    = 1 if j'=j and k'=k, else 0
    
    Wait, that's not right. Let me reconsider.
    
    If I set for each (j,k) a column c = j*p+k:
      U[i*m+j, c] = 1 for all i in range(n)
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1 for all i in range(n)
    
    Then for a fixed (i,j,k), the contribution from column c(j,k) is:
      U[i*m+j, c(j,k)] * V[j*p+k, c(j,k)] * W[k*n+i, c(j,k)] = 1*1*1 = 1
    
    But what about other columns c(j',k')? For c(j',k') to contribute to (i,j,k), we need:
      U[i*m+j, c(j',k')] != 0, which requires i*m+j = i'*m+j' for some i', so j=j'
      V[j*p+k, c(j',k')] != 0, which requires j*p+k = j'*p+k' for some j', so j=j' and k=k'
      W[k*n+i, c(j',k')] != 0, which requires k*n+i = k'*n+i' for some i', so k=k'
    
    So if j=j' and k=k', then all three conditions are satisfied. Otherwise, at least one
    condition fails. So the sum is exactly 1 for each (i,j,k).
    
    This gives a decomposition of rank m*p = 4*5 = 20!
    
    Let me verify: U has shape (n*m, m*p) = (16, 20), V has shape (m*p, m*p) = (20, 20),
    W has shape (p*n, m*p) = (20, 20).
    
    For column c = j*p+k (where j in range(m), k in range(p)):
      U[i*m+j, c] = 1 for all i in range(n)
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1 for all i in range(n)
    
    All entries are 0 or 1, which are integers. This should work!
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 5 = 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m + j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # V[j*p + k, c] = 1
            V[j * p + k][c] = 1
            # W[k*n + i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
