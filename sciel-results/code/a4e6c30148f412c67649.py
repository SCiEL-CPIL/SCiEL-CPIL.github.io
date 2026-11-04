import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_std = n * m * p  # 18
    
    # Build standard rank-18 decomposition
    U_std = [[0] * rank_std for _ in range(n * m)]
    V_std = [[0] * rank_std for _ in range(m * p)]
    W_std = [[0] * rank_std for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j][col] = 1
                V_std[j * p + k][col] = 1
                W_std[k * n + i][col] = 1
                col += 1
    
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # First, try to construct a rank-17 solution using Strassen-like decomposition
    # For 2x3x3, we can use Strassen on the 2x2x2 block and direct for remaining
    # This gives rank 7 (Strassen for 2x2x2) + 5 (direct for remaining) = 12? Let me think...
    # Actually, let's try a different approach: use the known rank-17 construction
    
    # Rank-17 construction based on Strassen's algorithm for 2x2x2 sub-block
    # The 2x3x3 multiplication can be decomposed as:
    # - 2x2x2 block (top-left): use Strassen (rank 7)
    # - 2x1x3 block (bottom-left): direct (rank 3)  
    # - 1x2x3 block (top-right): direct (rank 3)
    # - 1x1x3 block (bottom-right): direct (rank 3)
    # But this doesn't quite work due to overlapping indices
    
    # Instead, let's use a more systematic approach:
    # We know the theoretical lower bound is 15, and rank 17 is achievable
    # Let's try to find it via optimization with better initialization
    
    best_solution = None
    best_rank = 18
    best_err = float('inf')
    
    # Try ranks from 17 down to 16
    for target_rank in [17, 16]:
        if time.time() > deadline:
            break
        
        U_arr = np.array(U_std, dtype=float)
        V_arr = np.array(V_std, dtype=float)
        W_arr = np.array(W_std, dtype=float)
        
        def objective(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E = T_flat - T_rec.flatten()
            return np.dot(E, E)
        
        def gradient(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E_tensor = (T - T_rec)
            
            grad_U = -2 * np.einsum('abc,bt,ct->at', E_tensor, V, W)
            grad_V = -2 * np.einsum('abc,at,ct->bt', E_tensor, U, W)
            grad_W = -2 * np.einsum('abc,at,bt->ct', E_tensor, U, V)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        local_best_err = float('inf')
        local_best_params = None
        
        # Multiple restart strategies with better diversity
        num_restarts = 30 if target_rank == 17 else 20
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 17 + 42)
            
            # Strategy 1: Near-standard initialization with structured perturbation
            if restart < num_restarts // 3:
                U_init = np.zeros((dim_u, target_rank))
                V_init = np.zeros((dim_v, target_rank))
                W_init = np.zeros((dim_w, target_rank))
                
                # Copy first target_rank-1 columns from standard
                for t in range(min(target_rank - 1, 18)):
                    if t < 18:
                        U_init[:, t] = U_arr[:, t]
                        V_init[:, t] = V_arr[:, t]
                        W_init[:, t] = W_arr[:, t]
                
                # Last column: combine last few standard columns
                if target_rank - 1 < 18:
                    U_init[:, target_rank-1] = (U_arr[:, target_rank-1] + U_arr[:, min(target_rank, 17)]) / 2.0
                    V_init[:, target_rank-1] = (V_arr[:, target_rank-1] + V_arr[:, min(target_rank, 17)]) / 2.0
                    W_init[:, target_rank-1] = (W_arr[:, target_rank-1] + W_arr[:, min(target_rank, 17)]) / 2.0
                else:
                    U_init[:, target_rank-1] = np.mean(U_arr[:, -3:], axis=1)
                    V_init[:, target_rank-1] = np.mean(V_arr[:, -3:], axis=1)
                    W_init[:, target_rank-1] = np.mean(W_arr[:, -3:], axis=1)
                
                # Add structured perturbation
                scale = 0.05
                U_init += np.random.randn(dim_u, target_rank) * scale
                V_init += np.random.randn(dim_v, target_rank) * scale
                W_init += np.random.randn(dim_w, target_rank) * scale
                
            elif restart < 2 * num_restarts // 3:
                # Strategy 2: Random initialization with sparsity pattern
                U_init = np.random.randn(dim_u, target_rank) * 0.3
                V_init = np.random.randn(dim_v, target_rank) * 0.3
                W_init = np.random.randn(dim_w, target_rank) * 0.3
                
                # Enforce some sparsity
                for t in range(target_rank):
                    # Keep only a few non-zeros in each column
                    mask_u = np.random.choice(dim_u, size=max(1, dim_u // 3), replace=False)
                    mask_v = np.random.choice(dim_v, size=max(1, dim_v // 3), replace=False)
                    mask_w = np.random.choice(dim_w, size=max(1, dim_w // 3), replace=False)
                    
                    U_init[~np.isin(np.arange(dim_u), mask_u), t] = 0
                    V_init[~np.isin(np.arange(dim_v), mask_v), t] = 0
                    W_init[~np.isin(np.arange(dim_w), mask_w), t] = 0
                    
            else:
                # Strategy 3: Hybrid - start from standard and randomly merge columns
                U_init = np.zeros((dim_u, target_rank))
                V_init = np.zeros((dim_v, target_rank))
                W_init = np.zeros((dim_w, target_rank))
                
                # Randomly select and merge standard columns
                cols_to_use = list(range(18))
                np.random.shuffle(cols_to_use)
                
                for t in range(target_rank):
                    if t < len(cols_to_use):
                        col_idx = cols_to_use[t]
                        U_init[:, t] = U_arr[:, col_idx]
                        V_init[:, t] = V_arr[:, col_idx]
                        W_init[:, t] = W_arr[:, col_idx]
                    else:
                        # Average of two random standard columns
                        c1, c2 = np.random.choice(18, size=2, replace=False)
                        U_init[:, t] = (U_arr[:, c1] + U_arr[:, c2]) / 2.0
                        V_init[:, t] = (V_arr[:, c1] + V_arr[:, c2]) / 2.0
                        W_init[:, t] = (W_arr[:, c1] + W_arr[:, c2]) / 2.0
                
                # Add small perturbation
                U_init += np.random.randn(dim_u, target_rank) * 0.02
                V_init += np.random.randn(dim_v, target_rank) * 0.02
                W_init += np.random.randn(dim_w, target_rank) * 0.02
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            from scipy.optimize import minimize
            
            try:
                # Use L-BFGS-B with relaxed tolerances first, then tighten
                res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14, 'disp': False})
                
                if res.fun < local_best_err:
                    local_best_err = res.fun
                    local_best_params = res.x.copy()
                
                # If we found a very good solution, try to polish it
                if res.fun < 1e-8:
                    U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                    V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                    W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(2.0 * U_sol) / 2.0
                    V_rounded = np.round(2.0 * V_sol) / 2.0
                    W_rounded = np.round(2.0 * W_sol) / 2.0
                    
                    T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # Also try integer rounding
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    T_check_int = np.einsum('at,bt,ct->abc', U_int, V_int, W_int)
                    if np.allclose(T_check_int, T, atol=1e-9):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        
            except Exception:
                pass
        
        # If we found a good solution for this rank, try to polish it
        if local_best_err < 1e-6 and local_best_params is not None:
            U_sol = local_best_params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V_sol = local_best_params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W_sol = local_best_params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            
            # Polish with tighter tolerance
            params_polish = local_best_params.copy()
            from scipy.optimize import minimize
            try:
                res = minimize(objective, params_polish, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-16, 'disp': False})
                
                if res.fun < local_best_err:
                    local_best_err = res.fun
                    local_best_params = res.x.copy()
                    U_sol = res.x[0:dim_u*target_rank].reshape((dim_u, target_rank))
                    V_sol = res.x[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                    W_sol = res.x[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            except Exception:
                pass
            
            # Final check for exact half-integer solution
            U_rounded = np.round(2.0 * U_sol) / 2.0
            V_rounded = np.round(2.0 * V_sol) / 2.0
            W_rounded = np.round(2.0 * W_sol) / 2.0
            
            T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_check, T, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Try integer rounding as well
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            T_check_int = np.einsum('at,bt,ct->abc', U_int, V_int, W_int)
            if np.allclose(T_check_int, T, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
    
    # Return standard rank 18
    return {"U": U_std, "V": V_std, "W": W_std}
