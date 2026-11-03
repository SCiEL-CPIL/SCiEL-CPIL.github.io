import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_strassen = 28
    
    # 1. Build the Strassen solution (Rank 28)
    U_s = [[0] * rank_strassen for _ in range(n * m)]
    V_s = [[0] * rank_strassen for _ in range(m * p)]
    W_s = [[0] * rank_strassen for _ in range(p * n)]
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U_s[global_idx][col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V_s[global_idx][col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W_s[global_idx][col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    U_np_s = np.array(U_s, dtype=float)
    V_np_s = np.array(V_s, dtype=float)
    W_np_s = np.array(W_s, dtype=float)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    if a_col == b_row:
                        i = a_row * m + a_col
                        j = b_row * p + b_col
                        c_row = a_row
                        c_col = b_col
                        k = c_col * n + c_row
                        T_expected[i, j, k] = 1.0
                        
    # Time budget
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_rank = 28
    best_U, best_V, best_W = U_np_s, V_np_s, W_np_s
    
    target_rank = 27
    np.random.seed(42)
    
    def flatten_params(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        return U, V, W

    def loss(params):
        U, V, W = flatten_params(params)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = T - T_expected
        return np.sum(err**2)

    def loss_and_grad(params):
        U, V, W = flatten_params(params)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = T - T_expected
        loss_val = np.sum(err**2)
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', err, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', err, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', err, U, V)
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss_val, grad

    def create_initializations():
        inits = []
        
        # Strategy 1: Drop each column from Strassen, perturb
        for drop_col in range(rank_strassen):
            U_init = np.delete(U_np_s, drop_col, axis=1)
            V_init = np.delete(V_np_s, drop_col, axis=1)
            W_init = np.delete(W_np_s, drop_col, axis=1)
            
            noise_scale = 0.05
            U_init = U_init + noise_scale * np.random.randn(n*m, target_rank)
            V_init = V_init + noise_scale * np.random.randn(m*p, target_rank)
            W_init = W_init + noise_scale * np.random.randn(p*n, target_rank)
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            inits.append(params)
        
        # Strategy 2: Random linear combination of Strassen columns to form 27 columns
        for trial in range(20):
            # Pick a random 28x27 matrix M, form new columns as U_s @ M
            M = np.random.randn(rank_strassen, target_rank)
            # Scale to keep magnitudes similar
            M = M / np.linalg.norm(M, axis=0, keepdims=True) * np.sqrt(rank_strassen / target_rank)
            U_init = U_np_s @ M
            V_init = V_np_s @ M
            W_init = W_np_s @ M
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            inits.append(params)
        
        # Strategy 3: Pure random with scaling
        for _ in range(15):
            scale = np.random.uniform(0.5, 2.0)
            params = scale * np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank)
            inits.append(params)
        
        # Strategy 4: Block-aware: perturb within blocks
        for trial in range(10):
            U_init = U_np_s.copy()
            V_init = V_np_s.copy()
            W_init = W_np_s.copy()
            # Add random low-rank perturbation
            r_pert = np.random.randint(1, 4)
            U_init += np.random.randn(n*m, r_pert) @ np.random.randn(r_pert, rank_strassen)
            V_init += np.random.randn(m*p, r_pert) @ np.random.randn(r_pert, rank_strassen)
            W_init += np.random.randn(p*n, r_pert) @ np.random.randn(r_pert, rank_strassen)
            # Now project to rank 27 by keeping top 27 singular vectors
            # Use SVD on the combined tensor contribution
            U_init, sU, _ = np.linalg.svd(U_init, full_matrices=False)
            V_init, sV, _ = np.linalg.svd(V_init, full_matrices=False)
            W_init, sW, _ = np.linalg.svd(W_init, full_matrices=False)
            U_init = U_init[:, :target_rank]
            V_init = V_init[:, :target_rank]
            W_init = W_init[:, :target_rank]
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            inits.append(params)
        
        return inits

    # Two-stage optimization
    inits = create_initializations()
    
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            # Stage 1: Coarse optimization
            res1 = minimize(loss, x0, jac=loss_and_grad, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 2000, 'gtol': 1e-6, 'disp': False})
            
            if time.time() > deadline - 5:
                break
            
            # Check if close
            U_opt, V_opt, W_opt = flatten_params(res1.x)
            T_opt = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
            max_err = np.max(np.abs(T_opt - T_expected))
            
            if max_err < 1e-8:
                best_rank = 27
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                break
            
            # Stage 2: If close but not exact, do a polish
            if max_err < 1e-3:
                res2 = minimize(loss, res1.x, jac=loss_and_grad, method='L-BFGS-B', 
                              options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-12, 'disp': False})
                
                if time.time() > deadline:
                    break
                    
                U_opt2, V_opt2, W_opt2 = flatten_params(res2.x)
                T_opt2 = np.einsum('ir,jr,kr->ijk', U_opt2, V_opt2, W_opt2)
                max_err2 = np.max(np.abs(T_opt2 - T_expected))
                
                if max_err2 < 1e-8:
                    best_rank = 27
                    best_U, best_V, best_W = U_opt2, V_opt2, W_opt2
                    break
        except Exception:
            continue

    if best_rank == 27:
        def round_half(x):
            return np.round(x * 2) / 2.0
            
        U_final = round_half(best_U)
        V_final = round_half(best_V)
        W_final = round_half(best_W)
        
        T_check = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
        if np.max(np.abs(T_check - T_expected)) < 1e-9:
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        else:
            # Return unrounded if rounding fails
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Return Strassen (Rank 28)
    return {"U": U_np_s.tolist(), "V": V_np_s.tolist(), "W": W_np_s.tolist()}
# EVOLVE-BLOCK-END
