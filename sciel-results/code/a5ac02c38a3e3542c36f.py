import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def _compute_error_flat(params, n, m, p, rank):
    """Compute the squared error from flattened parameters."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_flat = params[:rank * nm]
    V_flat = params[rank * nm : rank * nm + rank * mp]
    W_flat = params[rank * nm + rank * mp : rank * nm + rank * mp + rank * pn]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = np.sum((T - 1.0) ** 2)
    return err

def _gradient_flat(params, n, m, p, rank):
    """Compute the gradient from flattened parameters."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_flat = params[:rank * nm]
    V_flat = params[rank * nm : rank * nm + rank * mp]
    W_flat = params[rank * nm + rank * mp : rank * nm + rank * mp + rank * pn]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    residual = T - 1.0
    
    dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', residual, V, W)
    dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', residual, U, W)
    dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', residual, U, V)
    
    dU_flat = dU.flatten()
    dV_flat = dV.flatten()
    dW_flat = dW.flatten()
    
    return np.concatenate([dU_flat, dV_flat, dW_flat])

def _optimize_rank(n, m, p, target_rank, max_iter=200, tol=1e-10, seed=0):
    """Optimize for a specific rank using scipy."""
    nm = n * m
    mp = m * p
    pn = p * n
    total_size = target_rank * (nm + mp + pn)
    
    rng = np.random.RandomState(seed)
    params0 = rng.randn(total_size) / np.sqrt(target_rank)
    
    def objective(params):
        return _compute_error_flat(params, n, m, p, target_rank)
    
    def gradient(params):
        return _gradient_flat(params, n, m, p, target_rank)
    
    try:
        res = minimize(objective, params0, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        final_err = res.fun
        if final_err < tol:
            U_flat = res.x[:target_rank * nm]
            V_flat = res.x[target_rank * nm : target_rank * nm + target_rank * mp]
            W_flat = res.x[target_rank * nm + target_rank * mp : target_rank * nm + target_rank * mp + target_rank * pn]
            
            U = U_flat.reshape(n * m, target_rank)
            V = V_flat.reshape(m * p, target_rank)
            W = W_flat.reshape(p * n, target_rank)
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W, final_err
        return None, None, None, final_err
    except Exception:
        return None, None, None, float('inf')

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    current_rank = n * m * p
    U_triv = np.zeros((n*m, current_rank))
    V_triv = np.zeros((m*p, current_rank))
    W_triv = np.zeros((p*n, current_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i*m + j, col] = 1
                V_triv[j*p + k, col] = 1
                W_triv[k*n + i, col] = 1
                col += 1
                
    best_rank = current_rank
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Search for lower rank
    # Start from a reasonable lower bound
    start_rank = 15
    
    for target_rank in range(start_rank, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            seed = attempt * 100 + target_rank
            U_opt, V_opt, W_opt, err = _optimize_rank(n, m, p, target_rank, max_iter=200, tol=1e-10, seed=seed)
            
            if U_opt is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
