"""Matrix multiplication 2x3 by 3x3: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
_START = time.time()
_DEADLINE = 145.0  # seconds, safety margin

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _verify(U, V, W):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.ones((n, m, p))
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    if not np.allclose(recon, T, atol=1e-8):
        return False
    return True

def _als_decompose(rank, n, m, p, max_iter=200, tol=1e-10, seed=0):
    """ALS optimization to find rank-r decomposition of the all-ones tensor."""
    rng = np.random.RandomState(seed)
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Build index lists
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        # Update U
        U_new = U.copy()
        for idx, (i, j, k) in enumerate(indices):
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            # Target: 1 = sum_c U[ui,c] * V[vi,c] * W[wi,c]
            # For fixed V, W: U[ui,:] should satisfy U[ui,:] @ (V[vi,:]*W[wi,:]) = 1
            # This is underdetermined, so use least squares with regularization
            vec = V[vi, :] * W[wi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            # Minimize ||U[ui,:] * vec - 1||^2 + lambda * ||U[ui,:]||^2
            lam = 1e-6
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            U_new[ui, :] = np.linalg.solve(A, b)
        U = U_new
        
        # Update V
        V_new = V.copy()
        for idx, (i, j, k) in enumerate(indices):
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            vec = U[ui, :] * W[wi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            lam = 1e-6
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            V_new[vi, :] = np.linalg.solve(A, b)
        V = V_new
        
        # Update W
        W_new = W.copy()
        for idx, (i, j, k) in enumerate(indices):
            ui, vi, wi = i * m + j, j * p + k, k * n + i
            vec = U[ui, :] * V[vi, :]
            if np.linalg.norm(vec) < 1e-12:
                continue
            lam = 1e-6
            A = np.outer(vec, vec) + lam * np.eye(rank)
            b = vec
            W_new[wi, :] = np.linalg.solve(A, b)
        W = W_new
        
        # Check convergence
        recon = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        err = np.max(np.abs(recon - 1.0))
        if err < tol:
            break
    
    # Final check
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-8:
        return U, V, W
    return None

def _try_rank(rank, n, m, p, max_restarts=50):
    """Try to find a rank-r decomposition with multiple restarts."""
    for seed in range(max_restarts):
        if _remaining() < 10:
            break
        result = _als_decompose(rank, n, m, p, max_iter=300, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return U, V, W
    return None

def _schoolbook():
    """Standard schoolbook decomposition, rank = n*m*p = 18."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return the rank-minimal exact tensor decomposition for 2x3 * 3x3.
    Search for lower-rank decompositions using ALS with multiple restarts.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 11 upward (theoretical lower bound is around 11-12)
    # Based on known results, the rank is likely 14 or 15 for 2x3x3
    for rank in range(11, 19):
        if _remaining() < 15:
            break
        result = _try_rank(rank, n, m, p, max_restarts=30)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook()
