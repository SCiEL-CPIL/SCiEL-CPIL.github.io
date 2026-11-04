"""Rank-minimal tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _build_target_tensor():
    """Build the matmul tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T, dim_u, dim_v, dim_w

def _validate_decomposition(U, V, W, T, dim_u, dim_v, dim_w):
    """Check if decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.einsum('at,bt,ct->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-8)

def _als_step_U(T, V, W, U, r):
    """Update U via ALS: for each t, solve linear system for U[:,t]."""
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    U_new = U.copy()
    for t in range(r):
        # Residual: R = T - sum_{t' != t} U[:,t'] outer V[:,t'] outer W[:,t']
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        # Subtract term t
        T_rec -= np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        R = T - T_rec
        # Solve: U[:,t] = argmin ||R - U[:,t] outer V[:,t] outer W[:,t]||
        # This is: minimize sum_{a} ||R[a,:,:] - U[a,t] * (V[:,t] outer W[:,t])||^2
        # For each a: U[a,t] = <R[a,:,:], V[:,t] outer W[:,t]> / ||V[:,t] outer W[:,t]||^2
        VW_outer = np.outer(V[:,t], W[:,t])
        denom = np.sum(VW_outer * VW_outer)
        if denom < 1e-15:
            continue
        for a in range(dim_u):
            U_new[a, t] = np.sum(R[a, :, :] * VW_outer) / denom
    return U_new

def _als_step_V(T, U, W, V, r):
    """Update V via ALS."""
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    V_new = V.copy()
    for t in range(r):
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        T_rec -= np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        R = T - T_rec
        UW_outer = np.outer(U[:,t], W[:,t])
        denom = np.sum(UW_outer * UW_outer)
        if denom < 1e-15:
            continue
        for b in range(dim_v):
            V_new[b, t] = np.sum(R[:, b, :] * UW_outer) / denom
    return V_new

def _als_step_W(T, U, V, W, r):
    """Update W via ALS."""
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    W_new = W.copy()
    for t in range(r):
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        T_rec -= np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        R = T - T_rec
        UV_outer = np.outer(U[:,t], V[:,t])
        denom = np.sum(UV_outer * UV_outer)
        if denom < 1e-15:
            continue
        for c in range(dim_w):
            W_new[c, t] = np.sum(R[:, :, c] * UV_outer) / denom
    return W_new

def _try_rank(T, dim_u, dim_v, dim_w, target_rank, deadline, n_restarts=50):
    """Try to find a decomposition of the given rank using ALS."""
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(restart * 13 + target_rank * 7)
        
        # Structured initialization: start from schoolbook and perturb
        # Schoolbook has rank 18; we need rank target_rank
        # Initialize with random values scaled appropriately
        U = np.random.randn(dim_u, target_rank) * 0.3
        V = np.random.randn(dim_v, target_rank) * 0.3
        W = np.random.randn(dim_w, target_rank) * 0.3
        
        # Run ALS for several iterations
        converged = False
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            
            U = _als_step_U(T, V, W, U, target_rank)
            V = _als_step_V(T, U, W, V, target_rank)
            W = _als_step_W(T, U, V, W, target_rank)
            
            # Check convergence
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.max(np.abs(T - T_rec))
            if err < 1e-10:
                converged = True
                break
        
        if not converged:
            continue
        
        # Try to round to half-integers
        U_res = np.round(2.0 * U) / 2.0
        V_res = np.round(2.0 * V) / 2.0
        W_res = np.round(2.0 * W) / 2.0
        
        if _validate_decomposition(U_res, V_res, W_res, T, dim_u, dim_v, dim_w):
            return (U_res, V_res, W_res)
        
        # If not close to half-integers, try a polish with smaller steps
        # and then round
        for polish in range(100):
            if time.time() > deadline - 2:
                break
            U = _als_step_U(T, V, W, U, target_rank)
            V = _als_step_V(T, U, W, V, target_rank)
            W = _als_step_W(T, U, V, W, target_rank)
        
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        if np.max(np.abs(T - T_rec)) < 1e-8:
            U_res = np.round(2.0 * U) / 2.0
            V_res = np.round(2.0 * V) / 2.0
            W_res = np.round(2.0 * W) / 2.0
            if _validate_decomposition(U_res, V_res, W_res, T, dim_u, dim_v, dim_w):
                return (U_res, V_res, W_res)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill
    
    T, dim_u, dim_v, dim_w = _build_target_tensor()
    
    # Schoolbook rank 18 as fallback
    rank_std = n * m * p
    U_std = np.zeros((dim_u, rank_std))
    V_std = np.zeros((dim_v, rank_std))
    W_std = np.zeros((dim_w, rank_std))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j, col] = 1
                V_std[j * p + k, col] = 1
                W_std[k * n + i, col] = 1
                col += 1
    
    # Try to find rank 17
    result = _try_rank(T, dim_u, dim_v, dim_w, 17, deadline, n_restarts=60)
    if result is not None:
        U_res, V_res, W_res = result
        return {
            "U": U_res.tolist(),
            "V": V_res.tolist(),
            "W": W_res.tolist()
        }
    
    # Return schoolbook
    return {
        "U": U_std.tolist(),
        "V": V_std.tolist(),
        "W": W_std.tolist()
    }
