import time
import itertools
from fractions import Fraction

import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _residual_func(x, n, m, p, T):
    """
    Residual function for least_squares.
    x is the flattened [U, V, W].
    """
    num_U = n * m
    num_V = m * p
    num_W = p * n
    r = len(x) // (num_U + num_V + num_W)
    
    U_flat = x[:num_U * r]
    V_flat = x[num_U * r : num_U * r + num_V * r]
    W_flat = x[num_U * r + num_V * r :]
    
    U = U_flat.reshape(num_U, r)
    V = V_flat.reshape(num_V, r)
    W = W_flat.reshape(num_W, r)
    
    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = T_approx - T
    return diff.flatten()

def _optimize_decomposition(n, m, p, r, T, deadline, seed=42):
    """
    Optimize a rank-r decomposition using Levenberg-Marquardt.
    """
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    num_U = n * m
    num_V = m * p
    num_W = p * n
    total_vars = (num_U + num_V + num_W) * r
    
    x0 = rng.randn(total_vars) * 0.1
    
    try:
        result = least_squares(
            _residual_func, 
            x0, 
            args=(n, m, p, T),
            method='lm', # Levenberg-Marquardt
            max_nfev=1000, # Max function evaluations
            x_scale='jac',
            ftol=1e-12,
            gtol=1e-12,
            xtol=1e-12
        )
    except Exception as e:
        return None
    
    if time.time() > deadline:
        return None
    
    if result.success or result.cost < 1e-6:
        x_opt = result.x
        U_flat = x_opt[:num_U * r]
        V_flat = x_opt[num_U * r : num_U * r + num_V * r]
        W_flat = x_opt[num_U * r + num_V * r :]
        
        U = U_flat.reshape(num_U, r)
        V = V_flat.reshape(num_V, r)
        W = W_flat.reshape(num_W, r)
        
        return U, V, W, result.cost
    
    return None

def _snap_to_half_integers(U, V, W, tol=1e-6):
    """
    Try to snap values to nearest half-integer if close.
    """
    def snap(M):
        snapped = np.round(M * 2) / 2
        if np.allclose(M, snapped, atol=tol, rtol=0):
            return snapped
        return None
    
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    if U_s is not None and V_s is not None and W_s is not None:
        return True, U_s, V_s, W_s
    return False, U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if the decomposition exactly reproduces T."""
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)

def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Try to find a decomposition with rank smaller than n*m*p.
    """
    start = time.time()
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from 95 down to 90
    ranks_to_try = list(range(best_rank - 1, best_rank - 6, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            res = _optimize_decomposition(n, m, p, r, T, deadline, seed=42 + attempt)
            
            if res is None:
                continue
            
            U, V, W, cost = res
            
            # Check if exact
            if _verify_decomposition(U, V, W, T, tol=1e-6):
                # Try to snap to half-integers
                snapped, U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if snapped:
                    if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-6):
                        best_rank = r
                        best_decomp = {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        found = True
                        break
                else:
                    # Use the float values converted to list
                    best_rank = r
                    best_decomp = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
        
        if found:
            # Continue trying for even smaller rank
            continue
    
    return best_rank, best_decomp

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 40s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    return decomp
# EVOLVE-BLOCK-END
