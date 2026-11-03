"""
Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
Uses a structured algebraic approach combined with numerical optimization.
"""
import time
import numpy as np
from scipy import linalg

_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s limit

def _time_remaining():
    return _TIMEOUT - (time.time() - _START_TIME)

def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p = 48."""
    n, m, p = 3, 4, 4
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
    return U, V, W

def _compute_tensor(U, V, W, n=3, m=4, p=4):
    """Compute T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _error(T):
    """Max absolute error from all-ones tensor."""
    return np.max(np.abs(T - 1.0))

def _als_step(U, V, W, n=3, m=4, p=4):
    """
    One ALS iteration solving for each factor.
    """
    rank = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Solve for U
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                sol, _, _, _ = linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i * m + j, :] = sol
            except:
                pass
    
    # Solve for V
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                sol, _, _, _ = linalg.lstsq(A, np.ones(n), rcond=None)
                V_new[j * p + k, :] = sol
            except:
                pass
    
    # Solve for W
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                sol, _, _, _ = linalg.lstsq(A, np.ones(m), rcond=None)
                W_new[k * n + i, :] = sol
            except:
                pass
    
    return U_new, V_new, W_new

def _als_optimize(U, V, W, n=3, m=4, p=4, max_iter=200, tol=1e-14):
    """Run ALS optimization with early stopping."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(max_iter):
        if _time_remaining() < 5:
            break
        
        U, V, W = _als_step(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _build_winograd_like(rank, n=3, m=4, p=4):
    """
    Build a Winograd-like structured initialization.
    For small matrices, we can exploit algebraic structure.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try a pattern based on tensor product structure
    # For 3x4*4x4, think of it as combining smaller multiplications
    
    # Initialize with a structured pattern
    np.random.seed(42)
    
    for r in range(rank):
        # Assign each rank term to a specific (i,j,k) pattern
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        # Base assignment
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
        
        # Add some cross-terms for sharing
        if r < rank // 2:
            i2 = (i + 1) % n
            j2 = (j + 1) % m
            k2 = (k + 1) % p
            U[i2 * m + j2, r] += 0.5
            V[j2 * p + k2, r] += 0.5
            W[k2 * n + i2, r] += 0.5
    
    return U, V, W

def _random_init(rank, n=3, m=4, p=4, seed=0):
    """Random initialization with scaling."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    scale = 1.0 / np.sqrt(rank)
    U *= scale
    V *= scale
    W *= scale
    return U, V, W

def _try_rank(target_rank, n=3, m=4, p=4, num_restarts=10):
    """Try to find exact decomposition at target rank."""
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        
        if restart % 3 == 0:
            U, V, W = _build_winograd_like(target_rank, n, m, p)
        elif restart % 3 == 1:
            U, V, W = _random_init(target_rank, n, m, p, seed=100 + restart)
        else:
            # Sparse initialization
            U, V, W = _build_winograd_like(target_rank, n, m, p)
            # Add random perturbations
            np.random.seed(200 + restart)
            U += 0.1 * np.random.randn(n * m, target_rank)
            V += 0.1 * np.random.randn(m * p, target_rank)
            W += 0.1 * np.random.randn(p * n, target_rank)
        
        U_sol, V_sol, W_sol, err = _als_optimize(U, V, W, n, m, p, max_iter=150)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_sol, V_sol, W_sol
        
        if err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def _polish(U, V, W, n=3, m=4, p=4):
    """High-precision polish."""
    if _time_remaining() < 5:
        return U, V, W, _error(_compute_tensor(U, V, W, n, m, p))
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _error(_compute_tensor(U, V, W, n, m, p))
    
    for it in range(100):
        if _time_remaining() < 3:
            break
        
        U, V, W = _als_step(U, V, W, n, m, p)
        T = _compute_tensor(U, V, W, n, m, p)
        err = _error(T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-15:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = 3, 4, 4
    max_rank = n * m * p  # 48
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = max_rank
    
    # Try ranks from low to high
    # Lower bounds: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Practical lower bound is higher; try from 24 upward
    for target_rank in range(24, max_rank):
        if _time_remaining() < 15:
            break
        
        num_restarts = max(5, min(15, int(_time_remaining() / 8)))
        
        U_try, V_try, W_try, err = _try_rank(target_rank, n, m, p, num_restarts)
        
        if err < 1e-10:
            # Found exact or near-exact decomposition
            U_try, V_try, W_try, err = _polish(U_try, V_try, W_try, n, m, p)
            
            if err < 1e-10:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                # Continue trying lower ranks
                continue
        elif err < 1e-4:
            # Close - try more restarts
            if _time_remaining() > 20:
                U_try2, V_try2, W_try2, err2 = _try_rank(target_rank, n, m, p, num_restarts=10)
                if err2 < 1e-10:
                    U_try2, V_try2, W_try2, err2 = _polish(U_try2, V_try2, W_try2, n, m, p)
                    if err2 < 1e-10:
                        best_U, best_V, best_W = U_try2, V_try2, W_try2
                        best_rank = target_rank
                        continue
    
    # Convert to lists
    U_list = [[float(val) for val in row] for row in best_U.tolist()]
    V_list = [[float(val) for val in row] for row in best_V.tolist()]
    W_list = [[float(val) for val in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
