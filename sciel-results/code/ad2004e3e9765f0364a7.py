import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute Target Tensor
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    # Precompute Schoolbook Solution (Rank 48)
    R_school = n * m * p
    U_school = np.zeros((size_u, R_school))
    V_school = np.zeros((size_v, R_school))
    W_school = np.zeros((size_w, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def objective_and_grad(params, U, V, W):
        T = tensor_contraction(U, V, W)
        E = T - TARGET_TENSOR
        loss = np.sum(E**2)
        
        # Gradients
        # dL/dU = 2 * E contracted with V and W
        # E is (size_u, size_v, size_w)
        # V is (size_v, R), W is (size_w, R)
        # grad_U (size_u, R) = 2 * sum_{v,w} E[u,v,w] V[v,r] W[w,r]
        # Use einsum: 'uvw,vr,wr->ur'
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', E, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', E, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', E, U, V)
        
        return loss, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def adam_step(params, grads, m_cache, v_cache, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        m_cache = beta1 * m_cache + (1 - beta1) * grads
        v_cache = beta2 * v_cache + (1 - beta2) * (grads ** 2)
        m_hat = m_cache / (1 - beta1 ** t)
        v_hat = v_cache / (1 - beta2 ** t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        return params, m_cache, v_cache

    best_solution = None
    best_rank = R_school
    best_params = None
    
    # Fallback
    fallback = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Search for lower ranks
    ranks_to_try = list(range(47, 24, -1)) # Down to 25
    
    for R in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Determine number of restarts and steps based on time
        time_left = deadline - time.time()
        if time_left > 60:
            n_restarts = 5
            n_steps = 2000
        elif time_left > 30:
            n_restarts = 3
            n_steps = 1000
        else:
            n_restarts = 2
            n_steps = 500
            
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R * 100 + restart)
            
            # Initialization
            # Strategy: Random Gaussian with small scale
            # Or: Random subset of schoolbook? 
            # Let's try Random Gaussian first, it's more general.
            params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.1
            
            U = params[:size_u * R].reshape(size_u, R)
            V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
            W = params[size_u * R + size_v * R :].reshape(size_w, R)
            
            m_cache = np.zeros_like(params)
            v_cache = np.zeros_like(params)
            
            best_loss_in_restart = float('inf')
            best_params_in_restart = params.copy()
            
            for step in range(n_steps):
                if time.time() > deadline - 5:
                    break
                    
                loss, grads = objective_and_grad(params, U, V, W)
                
                if loss < best_loss_in_restart:
                    best_loss_in_restart = loss
                    best_params_in_restart = params.copy()
                    
                if loss < 1e-10:
                    break
                    
                params, m_cache, v_cache = adam_step(params, grads, m_cache, v_cache, step + 1, lr=0.05)
                
                U = params[:size_u * R].reshape(size_u, R)
                V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W = params[size_u * R + size_v * R :].reshape(size_w, R)
                
            # Check best from this restart
            if best_loss_in_restart < 1e-8:
                U_opt = best_params_in_restart[:size_u * R].reshape(size_u, R)
                V_opt = best_params_in_restart[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W_opt = best_params_in_restart[size_u * R + size_v * R :].reshape(size_w, R)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify snapped solution
                T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                    best_rank = R
                    best_solution = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    break # Break restart loop, try next rank
            
    if best_solution:
        return best_solution
    else:
        return fallback

# EVOLVE-BLOCK-END
