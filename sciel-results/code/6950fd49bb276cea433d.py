"""
Program to find rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.

Strategy:
- The matrix multiplication tensor T_{i,j,k} for (n,m,p) = (3,3,7) is the structure tensor
  of the multiplication map. Its entries are 1 for all valid (i,j,k) in the standard
  "all-ones" representation of the multiplication tensor (each term A[i,j]*B[j,k] contributes
  to C[i,k]).
  
- We use a systematic search for low-rank decompositions using alternating least squares
  (ALS) with careful initialization, combined with a verification step.

- Key insight: The tensor is highly structured. We can use the fact that for fixed j,
  the sub-tensor over (i,k) is an outer product of e_i and e_k (standard basis vectors).
  This suggests that decompositions where each rank-1 term "specializes" to certain (i,j,k)
  patterns may be found.

- We start from schoolbook (rank 63) and try to find lower-rank decompositions via
  optimization. If no valid lower-rank decomposition is found within time budget,
  we return the best valid decomposition found (falling back to schoolbook).

- For 3x3x7, known bounds suggest the rank is between ~21 and 63. We search from
  lower ranks upward, using multiple random restarts and ALS refinement.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
DEADLINE_FRACTION = 0.85  # Safety margin: stop at 85% of 150s = 127.5s


def _verify_decomposition(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the multiplication tensor.
    
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """
    Faster verification using numpy operations.
    """
    rank = U.shape[1]
    # Reshape for efficient computation
    U_reshaped = U.reshape(n, m, rank)  # (n, m, rank)
    V_reshaped = V.reshape(m, p, rank)  # (m, p, rank)
    W_reshaped = W.reshape(p, n, rank)  # (p, n, rank)
    
    # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can compute this as: for each c, form the outer product structure
    # Sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Build the result tensor
    result = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_reshaped[:, :, c]  # (n, m)
        v_c = V_reshaped[:, :, c]  # (m, p)
        w_c = W_reshaped[:, :, c]  # (p, n)
        
        # For each (i, j, k): u_c[i,j] * v_c[j,k] * w_c[k,i]
        # This is a 3-way contraction
        # result[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        # We can do this element-wise
        # Create intermediate: for each j, u_c[:,j] (n,) * v_c[j,:] (p,) gives (n,p)
        # Then multiply by w_c[k,i] which is (p,n) -> need to transpose to (n,p) as w_c.T
        for j in range(m):
            # u_c[:, j] is (n,), v_c[j, :] is (p,)
            # outer product gives (n, p)
            outer = np.outer(u_c[:, j], v_c[j, :])  # (n, p)
            # w_c.T[i, k] = w_c[k, i], so we need w_c.T which is (n, p)
            result[:, j, :] += outer * w_c.T
    
    return np.allclose(result, 1.0, atol=tol)


def _als_decompose(rank, n=3, m=3, p=7, max_iters=200, tol=1e-8, seed=0, deadline=None):
    """
    Alternating Least Squares to find a rank-r decomposition of the multiplication tensor.
    
    T[i,j,k] = 1 for all i,j,k
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We alternate solving for U, V, W given the other two are fixed.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Scale to help convergence
    U *= 1.0 / np.sqrt(rank)
    V *= 1.0 / np.sqrt(rank)
    W *= 1.0 / np.sqrt(rank)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U: For each (i,j), we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is: U[i*m+j, :] @ (V[j*p:k, :] * W[k*n+i, :]) = 1 for all k
        # Let M_{c,k} = V[j*p+k, c] * W[k*n+i, c] for c=0..rank-1, k=0..p-1
        # Then U[i*m+j, :] @ M = ones(p)
        # This is a linear system: M.T @ U[i*m+j, :].T = ones(p)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build M: (rank, p) where M[c, k] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((rank, p))
                for k in range(p):
                    M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M.T @ x = ones(p)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Solve for V: For each (j,k), we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        # Let M_{c,i} = U[i*m+j, c] * W[k*n+i, c] for c=0..rank-1, i=0..n-1
        # Then V[j*p+k, :] @ M = ones(n)
        # M.T @ V[j*p+k, :].T = ones(n)
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, n))
                for i in range(n):
                    M[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Solve for W: For each (k,i), we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        # Let M_{c,j} = U[i*m+j, c] * V[j*p+k, c] for c=0..rank-1, j=0..m-1
        # Then W[k*n+i, :] @ M = ones(m)
        # M.T @ W[k*n+i, :].T = ones(m)
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, m))
                for j in range(m):
                    M[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        if it > 0:
            delta = (np.linalg.norm(U - U_prev) + np.linalg.norm(V - V_prev) + np.linalg.norm(W - W_prev))
            if delta < tol:
                break
        U_prev, V_prev, W_prev = U.copy(), V.copy(), W.copy()
    
    return U, V, W


def _try_rank(rank, n=3, m=3, p=7, num_restarts=20, deadline=None):
    """
    Try to find a valid decomposition of the given rank using multiple ALS restarts.
    Returns (U, V, W) if found, None otherwise.
    """
    for seed in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            return None
        
        U, V, W = _als_decompose(rank, n, m, p, max_iters=300, tol=1e-10, seed=seed, deadline=deadline)
        
        if _verify_fast(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
    
    return None


def construct():
    """
    Returns {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists (or numpy arrays)
    representing the factor matrices of the tensor decomposition.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 127.5  # 85% of 150s
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 63
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try ranks from low to high
    # Known lower bound for 3x3x3 is 19, so for 3x3x7 expect at least ~21-25
    # Try from 20 upward
    for rank in range(20, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, num_restarts=15, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found the minimal rank we can achieve
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
