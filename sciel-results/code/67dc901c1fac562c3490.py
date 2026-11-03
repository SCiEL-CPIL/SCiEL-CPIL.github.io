"""Rewritten program for matrix multiplication 3x5x7 rank minimization.
Uses global non-linear optimization to find a low-rank decomposition.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Create schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Note: The problem statement says W[k*n+i, r].
    # Let's check the indices carefully.
    # T[i,j,k] = 1.
    # Residual for (i,j,k) is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Dot product of rows U[u_idx], V[v_idx], W[w_idx] element-wise sum
                # This is not a standard dot product. It's sum_c U_c * V_c * W_c.
                prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                s = np.sum(prod)
                residuals.append(s - 1.0)
    return np.max(np.abs(residuals)) < 1e-9

def _residuals(params, n, m, p):
    """Calculate residuals for optimization.
    params: flat array of size R * (n*m + m*p + p*n)
    """
    R = len(params) // (n * m + m * p + p * n)
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = params[:size_u * R].reshape(size_u, R)
    V = params[size_u * R : (size_u + size_v) * R].reshape(size_v, R)
    W = params[(size_u + size_v) * R :].reshape(size_w, R)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                residuals.append(s - 1.0)
    return residuals

def _find_decomposition(n, m, p, max_rank=60, deadline=None):
    """Search for a decomposition of rank R using least_squares."""
    start_time = time.time()
    
    # Try ranks from low to high
    for R in range(20, max_rank + 1):
        if deadline and time.time() > deadline - 5:
            break
            
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_vars = R * (size_u + size_v + size_w)
        
        # Multiple random restarts for each rank
        for attempt in range(50):
            if deadline and time.time() > deadline - 2:
                break
                
            # Random initialization
            # Small random values to avoid huge numbers
            x0 = np.random.randn(total_vars) * 0.5
            
            try:
                res = least_squares(_residuals, x0, args=(n, m, p), method='trf', max_nfev=10000)
            except Exception:
                continue
                
            if res.success or np.max(np.abs(res.fun)) < 1e-9:
                # Reconstruct matrices
                U = res.x[:size_u * R].reshape(size_u, R)
                V = res.x[size_u * R : (size_u + size_v) * R].reshape(size_v, R)
                W = res.x[(size_u + size_v) * R :].reshape(size_w, R)
                
                # Exact verification
                if _verify(U, V, W, n, m, p):
                    return U, V, W, R
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p
    
    try:
        result = _find_decomposition(n, m, p, max_rank=80, deadline=deadline)
        if result:
            U, V, W, R = result
            if R < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = R
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
