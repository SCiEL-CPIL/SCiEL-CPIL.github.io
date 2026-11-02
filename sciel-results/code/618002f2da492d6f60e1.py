import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # 2. Try to find rank 41 using L-BFGS-B optimization with analytical gradients
    target_rank = 41
    A_dim = n * m  # 6
    B_dim = m * p  # 21
    C_dim = p * n  # 14
    
    def unpack(params):
        U = params[:A_dim*target_rank].reshape(A_dim, target_rank)
        V = params[A_dim*target_rank:A_dim*target_rank + B_dim*target_rank].reshape(B_dim, target_rank)
        W = params[A_dim*target_rank + B_dim*target_rank:].reshape(C_dim, target_rank)
        return U, V, W
    
    def loss_and_grad(params):
        U, V, W = unpack(params)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_rec - T_target
        loss = np.sum(diff**2)
        
        # Analytical gradients using einsum
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        return loss, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    for attempt in range(15):
        if time.time() > deadline - 25:
            break
        
        # Initialize with schoolbook minus one term
        remove_idx = (attempt % rank_schoolbook)
        U_init = np.delete(U_school, remove_idx, axis=1)
        V_init = np.delete(V_school, remove_idx, axis=1)
        W_init = np.delete(W_school, remove_idx, axis=1)
        
        # Add small noise to escape the exact schoolbook configuration
        noise_scale = 0.05
        U_init = U_init + noise_scale * np.random.randn(A_dim, target_rank)
        V_init = V_init + noise_scale * np.random.randn(B_dim, target_rank)
        W_init = W_init + noise_scale * np.random.randn(C_dim, target_rank)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B
        try:
            result = minimize(loss_and_grad, params_init, method='L-BFGS-B', 
                             options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-12})
        except Exception:
            continue
        
        params_opt = result.x
        U_opt, V_opt, W_opt = unpack(params_opt)
        
        # Project to half-integers
        U_half = np.round(2*U_opt)/2
        V_half = np.round(2*V_opt)/2
        W_half = np.round(2*W_opt)/2
        
        # Check exactness of projected solution
        T_rec_proj = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        if np.allclose(T_rec_proj, T_target, atol=1e-9):
            return {
                "U": U_half.tolist(),
                "V": V_half.tolist(),
                "W": W_half.tolist()
            }
        
        # Also check if the continuous solution is close to half-integers
        T_rec_cont = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
        if np.allclose(T_rec_cont, T_target, atol=1e-9):
            if (np.max(np.abs(2*U_opt - np.round(2*U_opt))) < 1e-4 and
                np.max(np.abs(2*V_opt - np.round(2*V_opt))) < 1e-4 and
                np.max(np.abs(2*W_opt - np.round(2*W_opt))) < 1e-4):
                U_final = np.round(2*U_opt)/2
                V_final = np.round(2*V_opt)/2
                W_final = np.round(2*W_opt)/2
                T_rec_final = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                if np.allclose(T_rec_final, T_target, atol=1e-9):
                    return {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
    
    return best_result
