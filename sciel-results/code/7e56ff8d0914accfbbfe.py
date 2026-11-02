"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """
    Verify U, V, W decomposition with exact half-integer arithmetic.
    """
    rank = U.shape[1]
    # Check if entries are close to half-integers
    U_int = (2 * U).round().astype(int)
    V_int = (2 * V).round().astype(int)
    W_int = (2 * W).round().astype(int)
    
    # Ensure rounding was safe
    if not np.allclose(U, U_int / 2.0, atol=1e-6): return False
    if not np.allclose(V, V_int / 2.0, atol=1e-6): return False
    if not np.allclose(W, W_int / 2.0, atol=1e-6): return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U_int[i * m + j][r] * V_int[j * p + k][r] * W_int[k * n + i][r]
                if total != 8:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
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

def _compute_loss_grad(params, n, m, p, rank):
    size_U = n * m * rank
    size_V = m * p * rank
    U = params[:size_U].reshape(n * m, rank)
    V = params[size_U:size_U + size_V].reshape(m * p, rank)
    W = params[size_U + size_V:].reshape(p * n, rank)
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                
                prod = ui * vj * wk
                s = np.sum(prod)
                diff = s - 1.0
                loss += diff * diff
                
                grad_U[i * m + j] += 2 * diff * vj * wk
                grad_V[j * p + k] += 2 * diff * ui * wk
                grad_W[k * n + i] += 2 * diff * ui * vj
                
    return loss, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

def _optimize_rank(r, n, m, p, num_attempts=10, timeout_per_attempt=5.0):
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    for attempt in range(num_attempts):
        start_t = time.time()
        if time.time() - start_t > timeout_per_attempt:
            break
            
        # Initialization: Sub-sample from schoolbook + noise
        if r <= 36:
            indices = np.random.choice(36, r, replace=False)
            U = U_sb[:, indices].copy()
            V = V_sb[:, indices].copy()
            W = W_sb[:, indices].copy()
            
            # Add noise
            noise_scale = 0.1
            U += np.random.normal(0, noise_scale, U.shape)
            V += np.random.normal(0, noise_scale, V.shape)
            W += np.random.normal(0, noise_scale, W.shape)
        else:
            U = np.random.normal(0, 0.1, (n*m, r))
            V = np.random.normal(0, 0.1, (m*p, r))
            W = np.random.normal(0, 0.1, (p*n, r))
            
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def loss_grad(params):
            return _compute_loss_grad(params, n, m, p, r)
            
        try:
            res = minimize(loss_grad, params0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10})
        except Exception:
            continue
            
        if res.fun < 1e-8:
            params = res.x
            size_U = n * m * r
            size_V = m * p * r
            U_opt = params[:size_U].reshape(n * m, r)
            V_opt = params[size_U:size_U + size_V].reshape(m * p, r)
            W_opt = params[size_U + size_V:].reshape(p * n, r)
            
            # Snap to half-integers
            U_snap = (2 * U_opt).round() / 2.0
            V_snap = (2 * V_opt).round() / 2.0
            W_snap = (2 * W_opt).round() / 2.0
            
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Try ranks from 35 down to 25
    for r in range(35, 24, -1):
        if time.time() > deadline - 10:
            break
            
        # Adjust attempts based on time remaining
        time_left = deadline - time.time()
        attempts = max(1, int(time_left / 5.0))
        
        res = _optimize_rank(r, n, m, p, num_attempts=attempts, timeout_per_attempt=4.0)
        if res is not None:
            return res
            
    # Fallback: Schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
