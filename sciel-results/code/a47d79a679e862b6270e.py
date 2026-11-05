"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _recover_factors(U, V, W, n, m, p, tol=1e-8):
    """
    Given factor matrices from SVD, snap to rational values and verify.
    U shape: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n+i, r]
    """
    rank = U.shape[1]
    # Snap to nearest 0.5
    U_s = np.round(2 * U) / 2
    V_s = np.round(2 * V) / 2
    W_s = np.round(2 * W) / 2
    
    # Verify the decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r U[i*m+j... wait, let me think about the indexing
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For fixed i,k: sum_{j,r} A[i,j]*B[j,k]*W[k*n+i, r]... 
    # Actually the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all i,j,k
    
    # Let's verify: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_s[i*m+j, r] * V_s[j*p+k, r] * W_s[k*n+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err < 1e-6:
        return U_s, V_s, W_s
    
    # Try integer snapping
    U_s2 = np.round(U)
    V_s2 = np.round(V)
    W_s2 = np.round(W)
    max_err2 = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_s2[i*m+j, r] * V_s2[j*p+k, r] * W_s2[k*n+i, r]
                max_err2 = max(max_err2, abs(s - 1.0))
    
    if max_err2 < 1e-6:
        return U_s2, V_s2, W_s2
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall clock limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor T[i,j,k] = 1
    T = _build_tensor(n, m, p)
    
    # The tensor T[i,j,k] = 1 for all i,j,k is a rank-1 tensor!
    # T[i,j,k] = 1*1*1 = U[i,j]*V[j,k]*W[k,i] where all are 1
    # So rank 1 suffices!
    
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # With rank 1: U[i*m+j, 0]=1, V[j*p+k, 0]=1, W[k*n+i, 0]=1
    # Product = 1*1*1 = 1 = T[i,j,k] ✓
    
    # But wait - is this the right convention? The output factor follows W[k*n+i, r]
    # This means the output C[k, i]... hmm, let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # For multiplying A (n×m) by B (m×p), the result C is (n×p).
    # But here it says W[k*n+i] which would be a p×n matrix if k indexes rows and i indexes cols.
    # That's C^T or the result is p×n?
    
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k], C is n×p.
    # The convention W[k*n+i] suggests output indexed as [k, i] with k in [0,p), i in [0,n)
    # So output is p×n, which is C^T.
    
    # The tensor for matrix multiplication: T[i,j,k] where
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 (the coefficient)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # With W indexed as W[k*n+i, r]
    
    # For rank 1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 ✓
    
    # This gives rank 1!
    rank = 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                assert s == 1, f"Failed at ({i},{j},{k}): got {s}"
    
    return {"U": U, "V": V, "W": W}
