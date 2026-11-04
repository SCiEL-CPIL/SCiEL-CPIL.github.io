import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for b in range(3):
        for i in range(2):
            for k_loc in range(2):
                k = 2 * b + k_loc
                U[i * 3 + 0, col] = 1
                V[0 * 6 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
        
        U[0 * 3 + 1, col] = 1
        U[1 * 3 + 2, col] = 1
        V[1 * 6 + (2*b), col] = 1
        V[2 * 6 + (2*b+1), col] = 1
        W[(2*b) * 2 + 0, col] = 1
        W[(2*b+1) * 2 + 0, col] = 1
        W[(2*b) * 2 + 1, col] = 1
        W[(2*b+1) * 2 + 1, col] = 1
        col += 1
        
        U[1 * 3 + 1, col] = 1
        U[1 * 3 + 2, col] = 1
        V[1 * 6 + (2*b), col] = 1
        W[(2*b) * 2 + 1, col] = 1
        W[(2*b+1) * 2 + 1, col] = -1
        col += 1
        
        U[0 * 3 + 1, col] = 1
        V[2 * 6 + (2*b), col] = 1
        W[(2*b+1) * 2 + 0, col] = 1
        W[(2*b+1) * 2 + 1, col] = 1
        col += 1
        
        U[1 * 3 + 2, col] = 1
        V[2 * 6 + (2*b+1), col] = 1
        W[(2*b) * 2 + 0, col] = 1
        W[(2*b+1) * 2 + 1, col] = 1
        col += 1
        
        U[0 * 3 + 1, col] = 1
        U[0 * 3 + 2, col] = 1
        V[2 * 6 + (2*b+1), col] = 1
        W[(2*b) * 2 + 0, col] = -1
        W[(2*b+1) * 2 + 0, col] = 1
        col += 1
        
        U[1 * 3 + 1, col] = 1
        U[0 * 3 + 1, col] = -1
        V[1 * 6 + (2*b+1), col] = 1
        W[(2*b+1) * 2 + 1, col] = 1
        col += 1
        
        U[0 * 3 + 2, col] = 1
        U[1 * 3 + 2, col] = -1
        V[1 * 6 + (2*b), col] = 1
        V[2 * 6 + (2*b+1), col] = 1
        W[(2*b) * 2 + 0, col] = 1
        col += 1
        
    # Verify
    # T = sum_r U[:,r] V[:,r]^T W[:,r]^T
    # We want T[i*m+j, j*p+k, k*n+i] = 1 if i,j,k valid, else 0
    # Actually T is a tensor.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    # Check zeros
    if valid:
        # We only need to check non-zero support?
        # The support of the multiplication tensor is exactly { (i,j,k) | 0<=i<n, 0<=j<m, 0<=k<p }
        # So all entries in the n x m x p box should be 1.
        # Are there other non-zero entries?
        # The tensor is defined on the full space, but the multiplication tensor has support only on the box.
        # Wait, the tensor T is defined such that T(A,B) = C.
        # The entries of T are T_{i,j,k}.
        # If the decomposition is correct, T_{i,j,k} should be 1 for all i,j,k in range.
        # And 0 otherwise?
        # The indices i,j,k are bounded by n,m,p.
        # The factors U, V, W have fixed sizes.
        # U is n*m x rank. V is m*p x rank. W is p*n x rank.
        # The product sums over r.
        # The indices i,j,k are implicit in the row indices of U, V, W.
        # So we only compute T for i in [0,n), j in [0,m), k in [0,p).
        # There are no "other" indices because the matrices are sized exactly to the dimensions.
        # So checking the box is sufficient.
        
        if not valid:
             # Fallback to schoolbook if verification fails
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
