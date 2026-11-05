import numpy as np
import time
from scipy.optimize import minimize
import sys

DIMENSIONS = [3, 3, 3]
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """
    Verify that the decomposition U, V, W correctly decomposes the 3x3 matrix multiplication tensor.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _check_half_integers(U, V, W, tol=1e-9):
    for M in [U, V, W]:
        for row in M:
            for x in row:
                if not _is_half_integer(x, tol):
                    return False
    return True

def _get_schoolbook(n=3, m=3, p=3):
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

def _round_to_half_ints(M, tol=1e-6):
    M_rounded = [[round(x * 2) / 2 for x in row] for row in M]
    M_np = np.array(M, dtype=float)
    M_round_np = np.array(M_rounded, dtype=float)
    if np.max(np.abs(M_np - M_round_np)) < tol:
        return M_rounded
    return None

def _optimize_rank(r, n=3, m=3, p=3, time_limit=30.0, n_restarts=5, seed=42):
    start_time = time.time()
    np.random.seed(seed)
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    def compute_loss_vec(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:2*n*m*r].reshape(m*p, r)
        W = x[2*n*m*r:].reshape(p*n, r)
        
        loss = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            loss += (val - 1.0) ** 2
        return loss

    def compute_loss_and_grad(x):
        # Simplified: use finite differences or just loss for L-BFGS-B (which uses numerical grad if not provided)
        # For speed, we provide just the loss. L-BFGS-B is robust.
        return compute_loss_vec(x)

    best_loss = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        x0 = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5
        
        # Optimization
        res = minimize(compute_loss_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_solution = res.x
            
            if best_loss < 1e-10:
                break
                
    if best_solution is not None and best_loss < 1e-8:
        x = best_solution
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:2*n*m*r].reshape(m*p, r)
        W = x[2*n*m*r:].reshape(p*n, r)
        
        U_r = _round_to_half_ints(U.tolist())
        V_r = _round_to_half_ints(V.tolist())
        W_r = _round_to_half_ints(W.tolist())
        
        if U_r is not None and V_r is not None and W_r is not None:
            if _verify_decomposition(U_r, V_r, W_r):
                if _check_half_integers(U_r, V_r, W_r):
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _find_decomposition(n=3, m=3, p=3, max_rank=27, time_limit=100.0):
    start_time = time.time()
    
    # Try ranks from 23 down to 27 (if 23 fails, try 24, etc.)
    # The optimal is 23.
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > time_limit - 5:
            break
        
        time_for_this_rank = min(25.0, time_limit - (time.time() - start_time))
        # Increase restarts for lower ranks?
        n_restarts = 10 if r == 23 else 5
        result = _optimize_rank(r, n, m, p, time_limit=time_for_this_rank, n_restarts=n_restarts)
        
        if result is not None:
            return result
    
    return _get_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    result = _find_decomposition(n, m, p, max_rank=27, time_limit=DEADLINE - SAFETY_MARGIN)
    
    if time.time() > deadline:
        return _get_schoolbook(n, m, p)
        
    return result
