import numpy as np
import time
import random
import scipy.optimize as opt

DIMENSIONS = [3, 3, 3]

def _schoolbook():
    n, m, p = 3, 3, 3
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_tensor():
    n, m, p = 3, 3, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i*m+j
                beta = j*p+k
                gamma = k*n+i
                T[alpha, beta, gamma] = 1
    return T

def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify(U, V, W, T=None):
    if T is None:
        T = _build_tensor()
    T_hat = _reconstruct(U, V, W)
    return np.allclose(T, T_hat, atol=1e-5)

def _search_rank(r, time_limit=30.0):
    n, m, p = 3, 3, 3
    T = _build_tensor()
    dim_u = n*m
    dim_v = m*p
    dim_w = p*n
    
    def objective_and_grad(x):
        U = x[0:dim_u*r].reshape(dim_u, r)
        V = x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
        W = x[dim_u*r+dim_v*r:].reshape(dim_w, r)
        
        T_hat = _reconstruct(U, V, W)
        diff = T_hat - T
        
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return np.sum(diff**2), grad

    start_time = time.time()
    best_val = float('inf')
    best_x = None
    
    # Try a few random restarts
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        x0 = np.random.randn(dim_u*r + dim_v*r + dim_w*r) * 0.1
        
        try:
            res = opt.minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                if best_val < 1e-4:
                    break
        except Exception:
            continue
            
    if best_x is not None and best_val < 1e-4:
        U = best_x[0:dim_u*r].reshape(dim_u, r)
        V = best_x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
        W = best_x[dim_u*r+dim_v*r:].reshape(dim_w, r)
        if _verify(U, V, W, T):
            return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 40s safety margin
    
    # Try to find lower rank
    # We know rank 23 is possible. Try 23 first, then 24, 25, 26.
    # If we find one, return it.
    for r in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _search_rank(r, time_limit=20.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
