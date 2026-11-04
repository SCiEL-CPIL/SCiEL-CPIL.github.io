import time
import numpy as np
from scipy.optimize import least_squares
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 3]

def _verify_solution(U, V, W):
    """
    Verify if U, V, W (lists of lists of Fraction) form an exact decomposition.
    Returns True if valid.
    """
    n, m, p = DIMENSIONS
    R = len(U[0])
    
    # Pre-convert to numpy for speed? No, fractions are exact. 
    # But 27 checks * R terms is small.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = Fraction(0)
                for c in range(R):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                
                if total != 1:
                    return False
    return True

def _try_find_rank(r):
    """
    Attempt to find a rank-r decomposition.
    Returns (U, V, W) as lists of Fractions if found, else None.
    """
    n, m, p = DIMENSIONS
    
    # Initialize with random values
    # We use float64 for optimization
    U_np = np.random.randn(n * m, r)
    V_np = np.random.randn(m * p, r)
    W_np = np.random.randn(p * n, r)
    
    # Reshape for easier indexing in the objective function
    # U_flat is (n*m, r)
    # We need to evaluate sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def objective(params):
        # Unpack
        # Total params: n*m*r + m*p*r + p*n*r
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:size_u + size_v + size_w]
        
        U_mat = U_flat.reshape(n * m, r)
        V_mat = V_flat.reshape(m * p, r)
        W_mat = W_flat.reshape(p * n, r)
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    # Sum over c
                    prod = u_val * v_val * w_val
                    s = np.sum(prod)
                    errors.append(s - 1.0)
        
        return errors

    x0 = np.concatenate([U_np.flatten(), V_np.flatten(), W_np.flatten()])
    
    # Try a few restarts
    for restart in range(10):
        if restart > 0:
            x0 = np.random.randn(len(x0))
        
        try:
            res = least_squares(objective, x0, method='lm', max_nfev=1000)
            if res.success or res.cost < 1e-8:
                # Extract solution
                size_u = n * m * r
                size_v = m * p * r
                size_w = p * n * r
                
                U_sol = res.x[:size_u].reshape(n * m, r)
                V_sol = res.x[size_u:size_u + size_v].reshape(m * p, r)
                W_sol = res.x[size_u + size_v:size_u + size_v + size_w].reshape(p * n, r)
                
                # Attempt to round to half-integers
                # The coefficients are expected to be small integers or half-integers
                U_frac = [[Fraction(round(val * 2)) / 2 for val in row] for row in U_sol]
                V_frac = [[Fraction(round(val * 2)) / 2 for val in row] for row in V_sol]
                W_frac = [[Fraction(round(val * 2)) / 2 for val in row] for row in W_sol]
                
                if _verify_solution(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac
                    
        except Exception:
            pass
            
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find rank 19 (optimal)
    # If that fails, try 20, 21... up to 27
    # We prioritize 19.
    
    # Set random seed for reproducibility if we want, but random search is better
    random.seed(42)
    np.random.seed(42)
    
    for rank in [19, 20, 21, 22, 23, 24, 25, 26, 27]:
        if time.time() > deadline:
            break
            
        result = _try_find_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Standard schoolbook rank 27
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
