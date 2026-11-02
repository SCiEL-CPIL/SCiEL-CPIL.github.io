"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import math
import numpy as np
import scipy.optimize as opt
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """
    Try to snap U, V, W to half-integers and verify.
    Returns (U_snapped, V_snapped, W_snapped) if valid, else None.
    """
    # First try snapping to nearest 0.5
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    
    if _verify(U_s, V_s, W_s, n, m, p, tol=1e-5):
        return U_s, V_s, W_s
    
    # Try snapping to nearest integer
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    if _verify(U_i, V_i, W_i, n, m, p, tol=1e-5):
        return U_i, V_i, W_i
        
    # Try a local search around the snapped values? 
    # For now, just return None. The caller can try to polish.
    return None

def _try_optimize(n, m, p, target_rank, time_limit, seed=42):
    """
    Use scipy.optimize.minimize to find a rank-`target_rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    R = target_rank
    
    def objective_full(params):
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_flat = params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = T - 1.0
        loss = np.sum(E * E)
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            
            dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU[:, r] = dU_r.flatten()
            dV[:, r] = dV_r.flatten()
            dW[:, r] = dW_r.flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    best_result = None
    best_loss = float('inf')
    
    # We will try multiple random initializations
    num_attempts = 8
    for attempt in range(num_attempts):
        if time.time() > start + time_limit - 5:
            break
            
        scale = 1.0 / math.sqrt(R)
        
        # Different scales for different attempts
        if attempt == 0:
            s = scale
        elif attempt == 1:
            s = 1.0
        elif attempt == 2:
            s = 0.5
        else:
            s = scale * (0.5 + attempt * 0.1)
            
        U0 = rng.randn(dim_U, R) * s
        V0 = rng.randn(dim_V, R) * s
        W0 = rng.randn(dim_W, R) * s
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Use L-BFGS-B with a generous number of iterations
            res = opt.minimize(objective_full, params0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        except Exception as e:
            continue
            
        if res.fun < best_loss:
            best_loss = res.fun
            if best_loss < 1e-8:
                U_sol = res.x[:dim_U * R].reshape(dim_U, R)
                V_sol = res.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
                W_sol = res.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
                
                snapped = _snap_and_verify(U_sol, V_sol, W_sol, n, m, p)
                if snapped:
                    return snapped
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Theoretical lower bound for 3x3x4 is likely 24 or 25.
    # We try from 24 upwards.
    # If we find a valid decomposition, we return it.
    
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 15:
            break
            
        seed = 42 + target_rank * 7
        result = _try_optimize(n, m, p, target_rank, remaining_time, seed=seed)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
                
    return best_result

# EVOLVE-BLOCK-END
