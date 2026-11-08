import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify(U, V, W, T):
    n, m, p = DIMENSIONS
    # Check if U, V, W are half-integers
    # Multiply by 2 and check if close to integer
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
    
    # Check exact decomposition
    # T_calc = sum_r U[:,r] V[:,r] W[:,r]
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_calc, T, atol=1e-8):
        return False
    return True

def _snap_to_half_integers(U, V, W):
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s

def _try_rank(R, T, time_limit):
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > time_limit:
            break
            
        # Initialize U, V, W randomly
        # Scale matters. Let's try small random values.
        U = np.random.randn(dim_u, R) * 0.5
        V = np.random.randn(dim_v, R) * 0.5
        W = np.random.randn(dim_w, R) * 0.5
        
        # Flatten
        def objective(x):
            u = x[:dim_u*R].reshape(dim_u, R)
            v = x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
            w = x[dim_u*R+dim_v*R:].reshape(dim_w, R)
            t_calc = np.einsum('ar,br,cr->abc', u, v, w)
            return (t_calc - T).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = least_squares(objective, x0, method='trf', max_nfev=1000)
            if res.success or res.cost < 1e-8:
                u_opt = res.x[:dim_u*R].reshape(dim_u, R)
                v_opt = res.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                w_opt = res.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                
                # Snap to half-integers
                u_s, v_s, w_s = _snap_to_half_integers(u_opt, v_opt, w_opt)
                
                if _verify(u_s, v_s, w_s, T):
                    return u_s, v_s, w_s
        except Exception:
            pass
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    T = _build_target_tensor()
    
    # Try to find lower rank
    # Schoolbook rank is 45
    for R in range(44, 30, -1):
        if time.time() > deadline:
            break
        result = _try_rank(R, T, deadline)
        if result is not None:
            U, V, W = result
            # Convert to list of lists as required by interface?
            # The interface says "factor matrices of shapes...". 
            # The seed returns list of lists.
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
