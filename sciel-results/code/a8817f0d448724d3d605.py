"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    # Use the known rank-7 bilinear algorithm for 3x3 matrix multiplication.
    # The 3x3x7 multiplication tensor decomposes into 7 independent 3x3x7
    # sub-tensors (one per column of the right factor), each using the same
    # 7-term identity.
    #
    # The 7-term algorithm for C = A*B (3x3):
    # w1 = A[0,0]*(B[0,0] - B[0,1])
    # w2 = A[0,0] + A[0,1]
    # w3 = A[1,1] + A[2,2]
    # w4 = A[2,2]*(B[1,1] - B[1,0])
    # w5 = (A[0,0]+A[1,1])*B[1,0]
    # w6 = (A[2,2]+A[2,1])*B[0,1]
    # w7 = (A[1,1]+A[2,2])*(B[2,0]+B[2,1]) + (A[0,0]+A[1,1])*(B[0,0]+B[0,1]) +
    #      A[0,0]*B[0,0] + A[1,1]*B[1,1] + A[2,2]*B[2,2]
    #
    # Actually, let me use a cleaner well-known rank-7 scheme.
    # For 3x3 multiplication, the known bilinear complexity is 19, not 7.
    # Wait - 3x3x7 refers to the TENSOR dimensions: n=3, m=3, p=7.
    # This means A is 3x3, B is 3x7, C is 3x7.
    # This is NOT standard matrix multiplication - it's a specific bilinear map.
    #
    # Let me re-examine: the tensor has entries T[a,b,c] where
    # a = i*m + j (i in [0,n), j in [0,m)) -> a in [0,9)
    # b = j*p + k (j in [0,m), k in [0,p)) -> b in [0,21)
    # c = k*n + i (k in [0,p), i in [0,n)) -> c in [0,21)
    # T[a,b,c] = 1 iff the (i,j) entry of A, (j,k) entry of B, and (k,i) entry of C
    # participate in the product.
    #
    # Actually for matrix multiplication A(3x3) * B(3x7) = C(3x7):
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # So T[a,b,c] = 1 when a=(i,j), b=(j,k), c=(k,i) for valid i,j,k.
    #
    # This is the standard matrix multiplication tensor with n=3, m=3, p=7.
    # The rank of 3x3x7 matrix multiplication... 
    #
    # Let me just use a structured decomposition. The trivial rank is 3*3*7=63.
    # But we can do better by exploiting the Kronecker product structure.
    #
    # Actually, let me think about this more carefully. The bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # with A in F^{3x3}, B in F^{3x7}, C in F^{3x7}.
    #
    # One approach: use the fact that this is like 7 independent 3x3x1 multiplications
    # (one per column of B), each of which has rank at most 9 (trivially) or 7 (using
    # the Strassen-like decomposition for 3x3x1).
    #
    # For 3x3x1: C[i] = sum_j A[i,j] B[j]
    # This is just a matrix-vector product, which has rank 9 trivially, but can be
    # done in rank 7? No, matrix-vector 3x3x1 has rank 9 (it's just a linear map).
    #
    # Wait, I'm confusing myself. Let me just construct a valid decomposition.
    #
    # The simplest valid decomposition is the trivial one with rank 63.
    # To improve, I'll use a block structure.
    
    # Trivial decomposition: rank = n*m*p = 63
    # Each term corresponds to one (i,j,k) triple
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
