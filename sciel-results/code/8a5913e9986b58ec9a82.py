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

def _newton_search(rank, n, m, p, seed, max_iter=100, tol=1e-8):
    """
    Attempt to find a rank-R decomposition using Newton's method.
    Returns (U, V, W) as lists of lists of floats if successful, else None.
    """
    rng = random.Random(seed)
    
    # Total number of variables
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    
    # Initialize with small random values
    x = np.array([rng.uniform(-1, 1) for _ in range(total_vars)])
    
    # Reshape for easier access
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    def get_tensors(x):
        u_flat = x[:num_u]
        v_flat = x[num_u:num_u+num_v]
        w_flat = x[num_u+num_v:]
        
        U = u_flat.reshape((n*m, rank))
        V = v_flat.reshape((m*p, rank))
        W = w_flat.reshape((p*n, rank))
        return U, V, W
    
    def compute_residuals(x):
        U, V, W = get_tensors(x)
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residuals.append(1.0 - s)
        return np.array(residuals)
    
    def compute_jacobian(x):
        U, V, W = get_tensors(x)
        J = np.zeros((n*m*p, total_vars))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k # Unique index for equation
                    # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                    for r in range(rank):
                        var_idx_u = (i*m+j)*rank + r
                        J[eq_idx, var_idx_u] = V[j*p+k, r] * W[k*n+i, r]
                        
                        var_idx_v = num_u + (j*p+k)*rank + r
                        J[eq_idx, var_idx_v] = U[i*m+j, r] * W[k*n+i, r]
                        
                        var_idx_w = num_u + num_v + (k*n+i)*rank + r
                        J[eq_idx, var_idx_w] = U[i*m+j, r] * V[j*p+k, r]
        return J
    
    # Newton Iteration
    for _ in range(max_iter):
        res = compute_residuals(x)
        if np.linalg.norm(res) < tol:
            break
        
        J = compute_jacobian(x)
        # Solve J delta = -res
        try:
            delta = np.linalg.lstsq(J, -res, rcond=None)[0]
        except Exception:
            break
            
        x = x + delta
        
    # Check final residual
    res = compute_residuals(x)
    if np.linalg.norm(res) < 1e-5:
        U, V, W = get_tensors(x)
        # Convert to lists
        U_list = [[float(v) for v in row] for row in U]
        V_list = [[float(v) for v in row] for row in V]
        W_list = [[float(v) for v in row] for row in W]
        return U_list, V_list, W_list
    return None

def _round_and_verify(U, V, W, n, m, p):
    """
    Round the float solution to nearest half-integers and verify.
    If it fails, try to correct via a local linear solve.
    """
    # Round to nearest 0.5
    U_r = [[round(v * 2) / 2.0 for v in row] for row in U]
    V_r = [[round(v * 2) / 2.0 for v in row] for row in V]
    W_r = [[round(v * 2) / 2.0 for v in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    
    # If rounding failed, try rounding to integers first?
    U_i = [[round(v) for v in row] for row in U]
    V_i = [[round(v) for v in row] for row in V]
    W_i = [[round(v) for v in row] for row in W]
    if _verify(U_i, V_i, W_i, n, m, p):
        return U_i, V_i, W_i
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook is always a valid fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Let's try ranks from 25 to 41
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random seeds for Newton's method
        for seed in range(10):
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
