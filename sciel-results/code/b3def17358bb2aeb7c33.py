import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # For matrix multiplication C = A @ B where A is n x m and B is m x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # Tensor decomposition: C = sum_r (sum_j U[i, j, r] A[i, j]) * (sum_{j,k} V[j, k, r] B[j, k])
    # But the standard form is:
    # C[i, k] = sum_r U[i, r] * V[r, k] where U and V are derived from A and B
    # 
    # More precisely, the bilinear form is:
    # C[i, k] = sum_{r} (sum_j U_{i,j,r} A_{i,j}) * (sum_{j',k'} V_{j',k',r} B_{j',k'})
    # 
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # T_{i,k,j} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where T_{i,k,j} = delta_{i,k} * A_{i,j} * B_{j,k}... no that's not right.
    # 
    # The multiplication tensor T has T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Actually, the tensor is T[i, k, j] = 1 for the term A[i,j]*B[j,k]
    # 
    # The decomposition is: T[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the interface.
    # 
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U has shape (n*m, rank) - indexed by (i,j)
    # V has shape (m*p, rank) - indexed by (j,k)
    # W has shape (p*n, rank) - indexed by (k,i)
    # 
    # The output factor follows the official trace convention W[k*n+i, r]
    # 
    # The decomposition should satisfy:
    # For the matrix multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) * (something)
    # 
    # Actually, the standard form is that the product of the three vectors for each r
    # gives the rank-1 tensor, and the sum over r gives the multiplication tensor.
    # 
    # The multiplication tensor for C = A@B where A is n x m, B is m x p, C is n x p:
    # T[i, k, j] = 1 if this term contributes to C[i,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # So the tensor is T with T[i, k, j] = 1 for all valid i, k, j
    # 
    # The decomposition: T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Wait, that doesn't make sense dimensionally. Let me think again.
    # 
    # In tensor decomposition for bilinear maps, we have:
    # f(A, B) = sum_r (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k])
    # and the result is a matrix C where C[i,k] = sum_r ... 
    # 
    # Actually, the standard way: the multiplication tensor T is in U^* x V^* x W^*
    # where U is the space of A (dim n*m), V is the space of B (dim m*p), W is the space of C (dim n*p)
    # 
    # T(a, b, c) = sum_{i,j,k} c[i,k] * T[i,k,j] * a[i,j] * b[j,k]
    # 
    # The rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in U^*, v_r in V^*, w_r in W^*
    # 
    # So T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For the multiplication tensor, T[i,k,j] = 1 for all i, j, k (since every term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, no. T[i,k,j] should be 1 if the term A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    # 
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # 
    # The trivial decomposition uses rank = n*m*p with U[i*m+j, r] = 1 if r corresponds to (i,j,k), etc.
    # 
    # For a better decomposition, we can try to find a lower rank.
    # 
    # For 2x3x8, let's try to use known results or heuristics.
    # 
    # One approach: since the matrix is 2x3, we can think of it as two 1x3 row vectors.
    # The multiplication can be seen as: for each row of A, compute the product with B.
    # 
    # Actually, let's just use the trivial decomposition for now, as finding the minimal rank
    # for 2x3x8 is non-trivial and I don't have specific knowledge of it.
    # 
    # The trivial rank is n*m*p = 2*3*8 = 48.
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
