import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    ijks = list(product(range(n), range(m), range(p)))
    U_idx = np.array([i*m + j for i,j,k in ijks])
    V_idx = np.array([j*p + k for i,j,k in ijks])
    W_idx = np.array([k*n + i for i,j,k in ijks])
    
    def compute_errors(U, V, W):
        R = U.shape[1]
        terms = U[U_idx] * V[V_idx] * W[W_idx]
        sums = terms.sum(axis=1)
        return np.abs(sums - 1.0)
    
    def verify(U, V, W):
        errors = compute_errors(U, V, W)
        return np.all(errors < 1e-9)
    
    # Schoolbook rank 100
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = np.zeros((n*m, schoolbook_rank))
    best_V = np.zeros((m*p, schoolbook_rank))
    best_W = np.zeros((p*n, schoolbook_rank))
    for idx, (i,j,k) in enumerate(ijks):
        best_U[i*m+j, idx] = 1.0
        best_V[j*p+k, idx] = 1.0
        best_W[k*n+i, idx] = 1.0
        
    # Try a rank-1 decomposition: the all-ones tensor has rank 1
    # For matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k in the standard definition
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = delta_{j, j} (identity in the middle)
    # Wait, the standard matrix multiplication tensor has T[i,j,k] = 1 if and only if the "middle" indices match
    # In the standard flattening: T[(i,j,k)] = 1 for all (i,j,k) where the middle index is the same
    # No, the matrix multiplication tensor is: for C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the i-j-k space)
    # Wait no. The tensor is T[i,j,k] = delta_{j,j} which is always 1. So it IS the all-ones tensor.
    
    # Let me reconsider. The tensor for matrix multiplication M_{n,m,p} is defined by:
    # T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    # This is because C[i,k] = sum_j A[i,j] * B[j,k], so each (i,j,k) term contributes with coefficient 1.
    # Therefore T is the all-ones tensor of shape (n, m, p).
    # The rank of the all-ones tensor of shape (n,m,p) is 1.
    
    # Let's try rank 1
    rank1_U = np.ones((n*m, 1))
    rank1_V = np.ones((m*p, 1))
    rank1_W = np.ones((p*n, 1))
    
    if verify(rank1_U, rank1_V, rank1_W):
        return {"U": rank1_U.tolist(), "V": rank1_V.tolist(), "W": rank1_W.tolist()}
    
    # If rank 1 doesn't work (which it shouldn't if my understanding is wrong),
    # fall back to schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
