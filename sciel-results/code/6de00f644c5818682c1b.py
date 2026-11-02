import time
import random
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

def _try_numerical_optimization(r, n, m, p, max_time, seed):
    """
    Attempts to find a rank-r decomposition using numerical optimization.
    Returns (U, V, W) if found, else None.
    """
    rng = random.Random(seed)
    
    # Flatten U, V, W into a single vector x
    # U: n*m x r
    # V: m*p x r
    # W: p*n x r
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    total_size = size_u + size_v + size_w
    
    def unpack(x):
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u+size_v]
        W_flat = x[size_u+size_v:]
        
        U = U_flat.reshape((n*m, r))
        V = V_flat.reshape((m*p, r))
        W = W_flat.reshape((p*n, r))
        return U, V, W

    def error_and_grad(x):
        U, V, W = unpack(x)
        
        # Compute S_ijk = sum_r U_ijr V_jkr W_kir
        # S is a 3D array of shape (n, m, p)
        # We can do this efficiently with einsum or loops
        # n=3, m=3, p=5, r is small (27-45)
        # Loops are fine for this size
        
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    S[i, j, k] = s
        
        # Error = sum (S - 1)^2
        err_vec = S - 1.0
        error = np.sum(err_vec ** 2)
        
        # Gradient
        # dE/dU_ijr = 2 * sum_k (S_ijk - 1) * V_jkr * W_kir
        # dE/dV_jkr = 2 * sum_i (S_ijk - 1) * U_ijr * W_kir
        # dE/dW_kir = 2 * sum_j (S_ijk - 1) * U_ijr * V_jkr
        
        grad_u = np.zeros_like(U)
        grad_v = np.zeros_like(V)
        grad_w = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dS = 2.0 * err_vec[i, j, k]
                    for rr in range(r):
                        grad_u[i*m+j, rr] += dS * V[j*p+k, rr] * W[k*n+i, rr]
                        grad_v[j*p+k, rr] += dS * U[i*m+j, rr] * W[k*n+i, rr]
                        grad_w[k*n+i, rr] += dS * U[i*m+j, rr] * V[j*p+k, rr]
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return error, grad

    # Try multiple restarts
    start_time = time.time()
    max_restarts = 20
    time_per_restart = max_time / max_restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        # Scale is important. 
        # If we start with 0, gradient is 0. 
        # Start with small random values.
        x0 = rng.uniform(-1, 1, total_size)
        
        # Optimize
        res = minimize(error_and_grad, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.success and res.fun < 1e-10:
            # Round to half-integers
            U, V, W = unpack(res.x)
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert to lists for validation
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _validate_half_integer(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
                
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 3, 5
    
    # Schoolbook fallback
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    
    best_rank = 45
    best_sol = schoolbook
    
    # Try to find a better rank using numerical optimization
    # Known rank for 3x3x5 is 27.
    # Try ranks from 27 upwards.
    
    for r in range(27, 45):
        if _time_remaining() < 15.0:
            break
            
        # Allocate more time to lower ranks
        time_budget = 30.0 if r == 27 else 10.0
        if _time_remaining() < time_budget:
            continue
            
        sol = _try_numerical_optimization(r, n, m, p, time_budget, seed=r*100)
        
        if sol is not None:
            U, V, W = sol
            best_rank = r
            best_sol = {"U": U, "V": V, "W": W}
            break # Found the minimum possible rank we tried
            
    return best_sol
