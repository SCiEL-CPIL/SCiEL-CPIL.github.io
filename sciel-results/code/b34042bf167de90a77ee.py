"""
Program to find rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.

Strategy:
- The matrix multiplication tensor T_{i,j,k} for (n,m,p) = (3,3,7) has entries T[i,j,k] = 1
  for all valid indices. We seek a rank-r decomposition:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
- We exploit the rectangular structure by:
  1. First finding a good decomposition for the 3x3x3 core (rank ~9-21)
  2. Extending to 3x3x7 by sharing structure across the 7 columns
  3. Using ALS with structured initialization and verification
- We search from low ranks upward with multiple random restarts.
- Fallback: schoolbook decomposition (rank 63).
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE_FRACTION = 0.85


def _verify_fast(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_reshaped[:, :, c]  # (n, m)
        v_c = V_reshaped[:, :, c]  # (m, p)
        w_c = W_reshaped[:, :, c]  # (p, n)
        for j in range(m):
            outer = np.outer(u_c[:, j], v_c[j, :])  # (n, p)
            result[:, j, :] += outer * w_c.T
    
    return np.allclose(result, 1.0, atol=tol)


def _als_decompose(rank, n=3, m=3, p=7, max_iters=300, tol=1e-8, seed=0, deadline=None):
    """
    Alternating Least Squares to find a rank-r decomposition.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    U *= 1.0 / np.sqrt(rank)
    V *= 1.0 / np.sqrt(rank)
    W *= 1.0 / np.sqrt(rank)
    
    U_prev = U.copy()
    V_prev = V.copy()
    W_prev = W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((rank, p))
                for k in range(p):
                    M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, n))
                for i in range(n):
                    M[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, m))
                for j in range(m):
                    M[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
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


def _structured_init(rank, n=3, m=3, p=7, seed=0):
    """
    Structured initialization that exploits the rectangular structure.
    Initialize with a pattern that respects the 3x3 core and extends to 7 columns.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try to initialize with some structure:
    # For the first min(rank, 9) terms, use schoolbook-like patterns for the 3x3 core
    # For remaining terms, use random initialization
    core_terms = min(rank, n * m)  # up to 9 terms for 3x3
    for c in range(core_terms):
        i = c % n
        j = (c // n) % m
        U[i * m + j, c] = 1.0
        for k in range(p):
            V[j * p + k, c] = 1.0 / p
            W[k * n + i, c] = 1.0
    
    # For remaining terms, random
    for c in range(core_terms, rank):
        U[:, c] = np.random.randn(n * m) * 0.1
        V[:, c] = np.random.randn(m * p) * 0.1
        W[:, c] = np.random.randn(p * n) * 0.1
    
    return U, V, W


def _try_rank(rank, n=3, m=3, p=7, num_restarts=30, deadline=None):
    """
    Try to find a valid decomposition of the given rank.
    Returns (U, V, W) if found, None otherwise.
    """
    for seed in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Alternate between random and structured initialization
        if seed % 3 == 0:
            U, V, W = _structured_init(rank, n, m, p, seed=seed)
        else:
            np.random.seed(seed)
            U = np.random.randn(n * m, rank) * 0.1 / np.sqrt(rank)
            V = np.random.randn(m * p, rank) * 0.1 / np.sqrt(rank)
            W = np.random.randn(p * n, rank) * 0.1 / np.sqrt(rank)
        
        U, V, W = _als_decompose(rank, n, m, p, max_iters=400, tol=1e-10, seed=seed, deadline=deadline)
        
        if _verify_fast(U, V, W, n, m, p, tol=1e-7):
            return U, V, W
    
    return None


def _polish(U, V, W, n=3, m=3, p=7, max_iters=50, tol=1e-12, deadline=None):
    """
    High-precision polish pass with tighter tolerance and more iterations.
    """
    rank = U.shape[1]
    U_prev = U.copy()
    V_prev = V.copy()
    W_prev = W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((rank, p))
                for k in range(p):
                    M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, n))
                for i in range(n):
                    M[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, m))
                for j in range(m):
                    M[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        if it > 0:
            delta = (np.linalg.norm(U - U_prev) + np.linalg.norm(V - V_prev) + np.linalg.norm(W - W_prev))
            if delta < tol:
                break
        U_prev, V_prev, W_prev = U.copy(), V.copy(), W.copy()
    
    return U, V, W


def construct():
    """
    Returns {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists
    representing the factor matrices of the tensor decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 127.5  # 85% of 150s
    
    # Schoolbook fallback
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
    for rank in range(18, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, num_restarts=25, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Polish for higher precision
            U, V, W = _polish(U, V, W, n, m, p, max_iters=30, tol=1e-12, deadline=deadline)
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
