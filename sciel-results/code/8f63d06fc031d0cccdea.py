"""Rewritten program for matrix multiplication tensor decomposition 3x3 by 3x4."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    def snap(x):
        return round(x * 2) / 2.0
    U_list = [[snap(x) for x in row] for row in U.tolist()]
    V_list = [[snap(x) for x in row] for row in V.tolist()]
    W_list = [[snap(x) for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False, 1.0
    error = np.max(np.abs(T - 1.0))
    return error < tol, error

def _compute_error(U_flat, V_flat, W_flat, n, m, p, rank):
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = T - 1.0
    return np.max(np.abs(err))

def _objective(params, n, m, p, rank, beta=10.0):
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.abs(T - 1.0)
    
    max_err = np.max(err)
    if max_err < 1e-12:
        return 0.0
    
    shifted = beta * err
    shifted_max = np.max(shifted)
    log_sum_exp = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
    soft_max = log_sum_exp / beta
    
    reg = 1e-6 * (np.sum(U_flat**2) + np.sum(V_flat**2) + np.sum(W_flat**2))
    return soft_max + reg

def _try_optimize(U_init, V_init, W_init, n, m, p, rank, deadline, max_time_per_attempt=5.0):
    U_flat = U_init.flatten()
    V_flat = V_init.flatten()
    W_flat = W_init.flatten()
    params = np.concatenate([U_flat, V_flat, W_flat])
    
    start_time = time.time()
    timeout = min(deadline - start_time, max_time_per_attempt)
    if timeout < 0.5:
        return None, None, None, 1.0
    
    try:
        result = minimize(
            _objective,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > timeout:
            return None, None, W_init, 1.0
        
        final_params = result.x
        U_opt = final_params[:n*m*rank].reshape(n * m, rank)
        V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
        
        err = _compute_error(U_opt.flatten(), V_opt.flatten(), W_opt.flatten(), n, m, p, rank)
        
        return U_opt, V_opt, W_opt, err
        
    except Exception:
        return None, None, None, 1.0

def _snap_and_verify(U, V, W, n, m, p):
    U_snap = np.round(U * 2) / 2.0
    V_snap = np.round(V * 2) / 2.0
    W_snap = np.round(W * 2) / 2.0
    valid, _ = _verify(U_snap, V_snap, W_snap, n, m, p)
    if valid:
        return U_snap, V_snap, W_snap
    return None, None, None

def _polish(U, V, W, n, m, p, deadline):
    if time.time() > deadline - 5:
        return None, None, None
    
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    params = np.concatenate([U_flat, V_flat, W_flat])
    rank = U.shape[1]
    
    try:
        result = minimize(
            _objective,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if time.time() > deadline - 3:
            return None, None, None
        
        final_params = result.x
        U_opt = final_params[:n*m*rank].reshape(n * m, rank)
        V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m * p, rank)
        W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p * n, rank)
        
        err = _compute_error(U_opt.flatten(), V_opt.flatten(), W_opt.flatten(), n, m, p, rank)
        
        if err < 1e-8:
            valid, _ = _verify(U_opt, V_opt, W_opt, n, m, p)
            if valid:
                return U_opt, V_opt, W_opt
        
        U_snap, V_snap, W_snap = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
        if U_snap is not None:
            return U_snap, V_snap, W_snap
            
    except Exception:
        pass
    
    return None, None, None

def _try_rank_fixed(U, V, W, n, m, p, rank, deadline, max_time_per_attempt=4.0):
    """Try to optimize a fixed-rank decomposition, with multiple strategies."""
    start = time.time()
    
    # Strategy 1: L-BFGS-B from current init
    U_opt, V_opt, W_opt, err = _try_optimize(U, V, W, n, m, p, rank, deadline, max_time_per_attempt)
    if U_opt is not None and err < 1e-8:
        valid, _ = _verify(U_opt, V_opt, W_opt, n, m, p)
        if valid:
            return U_opt, V_opt, W_opt
    if U_opt is not None and err < 0.5:
        U_snap, V_snap, W_snap = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
        if U_snap is not None:
            return U_snap, V_snap, W_snap
    if U_opt is not None and err < 0.1 and time.time() < deadline - 5:
        U_pol, V_pol, W_pol = _polish(U_opt, V_opt, W_opt, n, m, p, deadline)
        if U_pol is not None:
            return U_pol, V_pol, W_pol
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook baseline
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = _to_lists(U0, V0, W0)
    
    # Known lower bound: rank must be at least max(n*m, m*p, p*n) = 12
    # Upper bound: 36 (schoolbook)
    # Try ranks from 35 down to 12
    
    for target_rank in range(best_rank - 1, 11, -1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts with different scales
        for attempt in range(40):
            if time.time() > deadline - 10:
                break
            
            # Vary initialization strategy
            if attempt % 4 == 0:
                # Standard random
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
            elif attempt % 4 == 1:
                # Sparse random
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                # Zero out some entries
                mask = np.random.rand(*U.shape) < 0.3
                U[mask] = 0
            elif attempt % 4 == 2:
                # Binary-ish initialization
                U = np.random.choice([-1, 0, 1], size=(n * m, target_rank)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m * p, target_rank)).astype(float)
                W = np.random.choice([-1, 0, 1], size=(p * n, target_rank)).astype(float)
                # Ensure not all zeros
                if np.all(U == 0):
                    U[0, 0] = 1
                if np.all(V == 0):
                    V[0, 0] = 1
                if np.all(W == 0):
                    W[0, 0] = 1
            else:
                # Small perturbation around schoolbook-like structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Initialize with some structure
                for r in range(min(target_rank, 12)):
                    i = r % n
                    j = (r // n) % m
                    U[i * m + j, r] = 1.0
                    V[j * p + r % p, r] = 1.0
                    W[r % p * n + i, r] = 1.0
                # Add noise
                U += 0.1 * np.random.randn(n * m, target_rank)
                V += 0.1 * np.random.randn(m * p, target_rank)
                W += 0.1 * np.random.randn(p * n, target_rank)
            
            U_opt, V_opt, W_opt = _try_rank_fixed(U, V, W, n, m, p, target_rank, deadline)
            
            if U_opt is not None:
                best_rank = target_rank
                best_result = _to_lists(U_opt, V_opt, W_opt)
                return best_result
    
    # If we get here, try a few more ranks with more aggressive search
    for target_rank in [12, 13, 14, 15]:
        if time.time() > deadline - 5:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 3:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            U_opt, V_opt, W_opt = _try_rank_fixed(U, V, W, n, m, p, target_rank, deadline)
            
            if U_opt is not None:
                best_rank = target_rank
                best_result = _to_lists(U_opt, V_opt, W_opt)
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
