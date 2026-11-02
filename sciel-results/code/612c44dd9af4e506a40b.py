"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition_vectorized(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    results = np.ones((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = Ua[i*m+j, :]
                v_val = Va[j*p+k, :]
                w_val = Wa[k*n+i, :]
                results[i,j,k] = np.dot(u_val * v_val, w_val)
    
    return np.allclose(results, 1.0, atol=tol)

def _als_cp_decomposition(rank, n_iters=200, tol=1e-12, seed=0):
    """ALS for CP decomposition of the all-ones 3D tensor T (shape n×m×p).
    
    We want T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n, m, rank)
    V = rng.randn(m, p, rank)
    W = rng.randn(p, n, rank)
    
    # Normalize
    for r in range(rank):
        norm = np.linalg.norm(U[:, :, r])
        if norm > 0:
            U[:, :, r] /= norm
            V[:, :, r] *= norm
            W[:, :, r] *= norm
    
    for it in range(n_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((rank, p))
                for r in range(rank):
                    M[r, :] = V[j, :, r] * W[:, i, r]
                try:
                    U[i, j, :] = np.linalg.lstsq(M, T[i, j, :], rcond=None)[0]
                except:
                    U[i, j, :] = rng.randn(rank)
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, n))
                for r in range(rank):
                    M[r, :] = U[:, j, r] * W[k, :, r]
                try:
                    V[j, k, :] = np.linalg.lstsq(M, T[:, j, k], rcond=None)[0]
                except:
                    V[j, k, :] = rng.randn(rank)
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, m))
                for r in range(rank):
                    M[r, :] = U[i, :, r] * V[:, k, r]
                try:
                    W[k, i, :] = np.linalg.lstsq(M, T[i, :, k], rcond=None)[0]
                except:
                    W[k, i, :] = rng.randn(rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(U[i,j,:], V[j,k,:] * W[k,i,:])
                    err += (pred - 1.0)**2
        if err < tol:
            break
    
    # Convert to our format
    U_mat = [[0.0]*rank for _ in range(n*m)]
    V_mat = [[0.0]*rank for _ in range(m*p)]
    W_mat = [[0.0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            U_mat[i*m+j] = U[i,j,:].tolist()
    for j in range(m):
        for k in range(p):
            V_mat[j*p+k] = V[j,k,:].tolist()
    for k in range(p):
        for i in range(n):
            W_mat[k*n+i] = W[k,i,:].tolist()
    
    return U_mat, V_mat, W_mat

def _try_rank(rank, n_trials=20, deadline=None):
    """Try to find a valid rank-r decomposition."""
    n, m, p = DIMENSIONS
    best = None
    best_err = float('inf')
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        seed = trial * 1000 + 42
        U, V, W = _als_cp_decomposition(rank, n_iters=100, tol=1e-14, seed=seed)
        
        # Check error
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(Ua[i*m+j,:], Va[j*p+k,:] * Wa[k*n+i,:])
                    err += (pred - 1.0)**2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best = (U, V, W, err)
        
        if err < 1e-10:
            return best
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try ranks from low to high
    # Known: 4x4x4 has rank <= 20. For 4x4x6, let's try from 20 upward.
    for rank in range(20, 50):
        if time.time() > deadline - 20:
            break
        result = _try_rank(rank, n_trials=10, deadline=deadline - 20)
        if result is not None:
            U, V, W, err = result
            if err < 1e-10:
                # Verify exactly
                if _verify_decomposition_vectorized(U, V, W, n, m, p, tol=1e-9):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
