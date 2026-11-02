"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from scipy.optimize import minimize

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _snap_to_half_int(U, V, W):
    U = [[round(x * 2) / 2 for x in row] for row in U]
    V = [[round(x * 2) / 2 for x in row] for row in V]
    W = [[round(x * 2) / 2 for x in row] for row in W]
    return U, V, W

def _check_exact(U, V, W, T):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    R = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    err = np.sum((R - T)**2)
    return err < 1e-9

def _try_rank(n, m, p, rank, T, deadline, restarts=10):
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_shape = (nm, rank)
    V_shape = (mp, rank)
    W_shape = (pn, rank)
    
    size_u = nm * rank
    size_v = mp * rank
    size_w = pn * rank
    total_size = size_u + size_v + size_w
    
    def objective(x):
        U = x[:size_u].reshape(U_shape)
        V = x[size_u:size_u+size_v].reshape(V_shape)
        W = x[size_u+size_v:].reshape(W_shape)
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((R - T)**2)
        
    def callback(xk):
        if time.time() > deadline:
            raise ValueError("Timeout")

    for r in range(restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale: entries are roughly 1/sqrt(rank) or similar? 
        # Let's start with normal(0, 1)
        x0 = np.random.randn(total_size)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}, callback=callback)
        except ValueError:
            break
        except Exception:
            continue
            
        if res.success or (hasattr(res, 'fun') and res.fun < 1e-8):
            x_opt = res.x
            U = x_opt[:size_u].reshape(U_shape)
            V = x_opt[size_u:size_u+size_v].reshape(V_shape)
            W = x_opt[size_u+size_v:].reshape(W_shape)
            
            # Snap
            U_s, V_s, W_s = _snap_to_half_int(U.tolist(), V.tolist(), W.tolist())
            
            if _check_exact(U_s, V_s, W_s, T):
                return U_s, V_s, W_s
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from 24 down to 15
    # Schoolbook is 24. 
    # Known lower bounds suggest 18 is possible for 2x3x4? 
    # Let's try 20, 19, 18, 17, 16, 15
    for rank in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, rank, T, deadline, restarts=15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook rank 24
    rank = 24
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
# EVOLVE-BLOCK-END
