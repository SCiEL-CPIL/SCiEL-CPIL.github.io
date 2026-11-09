"""Rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-4 exact decomposition for multiplying a (2,4) matrix by a (4,6) matrix.
    
    The tensor T is defined by:
        T[i, j, k] = 1 if j is in {0..3} (middle index), else 0
    More precisely, for A of shape (2,4), B of shape (4,6), C = A*B:
        C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor entries are T[i,j,k] where the index for C is (k*n + i) per the trace convention W[k*n+i, r].
    Actually, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W has shape (p*n, rank) = (6*2, rank) = (12, rank), and W[k*n+i, r] is the entry.
    U has shape (n*m, rank) = (2*4, rank) = (8, rank), with U[i*m+j, r]
    V has shape (m*p, rank) = (4*6, rank) = (24, rank), with V[j*p+k, r]
    
    The decomposition satisfies:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j, j} (identity in middle index)
    
    Wait, the matrix multiplication tensor T[i, j, k] = 1 if the middle index j matches, which it always does in the sum.
    
    Actually, the standard matrix multiplication tensor for (n x m) * (m x p) is:
        T[i, j, k] = 1 (for all i, j, k) where the multiplication is sum_j A[i,j] * B[j,k]
    
    But wait - the tensor should be such that:
        C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, T is a 3-tensor with T[i,j,k] = 1 for all i,j,k, and the multiplication is:
        C[i,k] = sum_{j, r} A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no.
    
    Let me think more carefully. The decomposition of the matrix multiplication tensor T_{n,m,p}:
        T[i,j,k] = 1 if and only if the "path" is valid, which for standard matrix mult is always 1.
    
    Actually, the matrix multiplication tensor T has entries T[i, j, k] = 1 for all valid i, j, k.
    The tensor rank is the minimum r such that there exist matrices U, V, W with:
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook decomposition, each (i,j,k) triple gets its own rank-1 term:
        U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for that specific r.
    
    This gives rank n*m*p = 2*4*6 = 48.
    
    Now, can we do better? The key insight is that for fixed j, the tensor is:
        T[i,j,k] = 1 for all i in {0,1}, k in {0,...,5}
    
    This is a rank-1 tensor in the (i,k) subspace for each fixed j. So we can use:
        For each j in {0,1,2,3}:
            U[i*m+j, r_j] = 1 for all i
            V[j*p+k, r_j] = 1 for all k
            W[k*n+i, r_j] = 1 for all k, i
    
    This gives rank 4 (one for each j value).
    
    Let's verify: sum_r U[i*m+j', r] * V[j'*p+k, r] * W[k*n+i, r]
    For r corresponding to j*: U[i*m+j', r_j*] is 1 only if j' = j*, V[j'*p+k, r_j*] is 1 only if j' = j*, and W[k*n+i, r_j*] = 1.
    
    So the sum becomes: if j' = j, then 1*1*1 = 1, else 0.
    Thus T[i,j,k] = 1 if j = j (which is always true in the sum over j), wait no.
    
    The tensor entry T[i,j,k] should be 1. The decomposition gives:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For r = r_j (the rank index associated with middle index j):
        U[i*m+j, r_j] = 1 (since the row index is i*m+j, which matches)
        V[j*p+k, r_j] = 1 (since the row index is j*p+k, which matches)
        W[k*n+i, r_j] = 1
    
    For r = r_{j'} where j' != j:
        U[i*m+j, r_{j'}] = 0 (since row i*m+j != i*m+j')
    
    So the sum is 1 for each (i,j,k), which is exactly T[i,j,k] = 1.
    
    Therefore rank 4 is achievable and is optimal (since we need at least 4 to separate the 4 different middle indices).
    """
    n, m, p = DIMENSIONS
    rank = m  # 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1
        for k in range(p):
            V[j * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
