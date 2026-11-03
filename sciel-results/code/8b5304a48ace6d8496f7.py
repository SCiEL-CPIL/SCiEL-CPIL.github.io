import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Target: 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _fast_residuals(U, V, W, n, m, p, rank):
    """Compute residuals efficiently using numpy broadcasting."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    residuals = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residuals[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
    return 1.0 - residuals

def _newton_search(rank, n, m, p, seed, max_iter=50, tol=1e-10):
    """
    Attempt to find a rank-R decomposition using Newton's method with efficient Jacobian.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    rng = random.Random(seed)
    
    # Initialize with small random values
    U = np.random.uniform(-0.5, 0.5, (n*m, rank))
    V = np.random.uniform(-0.5, 0.5, (m*p, rank))
    W = np.random.uniform(-0.5, 0.5, (p*n, rank))
    
    for _ in range(max_iter):
        # Compute residuals
        res = _fast_residuals(U, V, W, n, m, p, rank)
        if np.linalg.norm(res) < tol:
            break
        
        # Compute Jacobian efficiently
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        total_vars = num_u + num_v + num_w
        
        J = np.zeros((n*m*p, total_vars))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    
                    for r in range(rank):
                        # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                        var_idx_u = (i*m+j)*rank + r
                        J[eq_idx, var_idx_u] = V[j*p+k, r] * W[k*n+i, r]
                        
                        # d/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                        var_idx_v = num_u + (j*p+k)*rank + r
                        J[eq_idx, var_idx_v] = U[i*m+j, r] * W[k*n+i, r]
                        
                        # d/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                        var_idx_w = num_u + num_v + (k*n+i)*rank + r
                        J[eq_idx, var_idx_w] = U[i*m+j, r] * V[j*p+k, r]
        
        # Solve J delta = -res.flatten()
        try:
            delta = np.linalg.lstsq(J, -res.flatten(), rcond=None)[0]
        except Exception:
            break
        
        # Update
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        U_flat += delta[:num_u]
        V_flat += delta[num_u:num_u+num_v]
        W_flat += delta[num_u+num_v:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
    
    # Check final residual
    res = _fast_residuals(U, V, W, n, m, p, rank)
    if np.linalg.norm(res) < 1e-6:
        return U, V, W
    return None

def _round_and_verify(U, V, W, n, m, p):
    """
    Round the float solution to nearest half-integers and verify.
    """
    # Round to nearest 0.5
    U_r = [[round(float(v) * 2) / 2.0 for v in row] for row in U]
    V_r = [[round(float(v) * 2) / 2.0 for v in row] for row in V]
    W_r = [[round(float(v) * 2) / 2.0 for v in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    
    # If rounding failed, try rounding to integers first?
    U_i = [[round(float(v)) for v in row] for row in U]
    V_i = [[round(float(v)) for v in row] for row in V]
    W_i = [[round(float(v)) for v in row] for row in W]
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i, V_i, W_i
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook is always a valid fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Try ranks from 22 to 41
    for rank in range(22, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random seeds for Newton's method
        for seed in range(5):
            if time.time() > deadline:
                break
                
            result = _newton_search(rank, n, m, p, seed)
            if result:
                U, V, W = result
                verified = _round_and_verify(U, V, W, n, m, p)
                if verified:
                    U_r, V_r, W_r = verified
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = rank
                    found = True
                    break # Found a valid decomposition for this rank, try lower rank
        
    return best
# EVOLVE-BLOCK-END
