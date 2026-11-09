import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        # Check conditions:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        # sum_r U[i,j1,r] V[j2,k,r] W[k,i,r] == 0 for j1 != j2
        
        # We can vectorize this check
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute the tensor T[i, j1, j2, k] = sum_r U[i,j1,r] V[j2,k,r] W[k,i,r]
        # Note: W index is k*n+i, so W_reshaped[k, i, r]
        
        # We need T[i, j, j, k] == 1 and T[i, j1, j2, k] == 0 for j1 != j2
        # Let's compute the full tensor or just check the conditions
        
        # Efficient check:
        # For each i, k, we need U_i^T (V_k \odot W_ik) = I_m ? No.
        # The condition is:
        # For fixed i, k:
        # Let W_ik be a vector of length R.
        # Let U_i be a matrix of shape (m, R).
        # Let V_k be a matrix of shape (R, m) (columns are v_{j,k}).
        # The term for (j1, j2) is sum_r U[i, j1, r] V[j2, k, r] W[k, i, r].
        # This is element (j1, j2) of U_i @ (V_k * W_ik[:, None])?
        # V_k is (R, m). V_k * W_ik[:, None] is (R, m).
        # U_i is (m, R).
        # Product is (m, m).
        # We need this product to be I_m.
        
        for i in range(n):
            for k in range(p):
                w_ik = W_reshaped[k, i, :]
                u_i = U_reshaped[i, :, :] # (m, R)
                v_k = V_reshaped[:, k, :] # (m, R) -> We need (R, m) for multiplication?
                # v_{j,k} is a vector. We want M[j1, j2] = sum_r u_{j1} v_{j2} w.
                # M = U_i @ (V_k * w_ik[:, None]).T ?
                # V_k is (m, R). V_k * w_ik[:, None] is (m, R).
                # (V_k * w_ik[:, None]).T is (R, m).
                # U_i is (m, R).
                # Product is (m, m).
                M = u_i @ (v_k * w_ik[:, None]).T
                # Check if M is close to I_m
                if not np.allclose(M, np.eye(m), atol=1e-5):
                    return False
        return True

    def error_func(params):
        R = len(params) // (dim_u + dim_v + dim_w)
        U_flat = params[:dim_u * R].reshape(dim_u, R)
        V_flat = params[dim_u * R : dim_u * R + dim_v * R].reshape(dim_v, R)
        W_flat = params[dim_u * R + dim_v * R :].reshape(dim_w, R)
        
        U_reshaped = U_flat.reshape(n, m, R)
        V_reshaped = V_flat.reshape(m, p, R)
        W_reshaped = W_flat.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                w_ik = W_reshaped[k, i, :]
                u_i = U_reshaped[i, :, :]
                v_k = V_reshaped[:, k, :]
                M = u_i @ (v_k * w_ik[:, None]).T
                err += np.sum((M - np.eye(m)) ** 2)
        return err

    # Start with schoolbook rank
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution
    U_sb = np.zeros((dim_u, best_rank))
    V_sb = np.zeros((dim_v, best_rank))
    W_sb = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to reduce rank
    for R in range(best_rank - 1, max(10, best_rank - 20), -1):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        np.random.seed(42)
        U_init = np.random.rand(dim_u, R) * 2 - 1
        V_init = np.random.rand(dim_v, R) * 2 - 1
        W_init = np.random.rand(dim_w, R) * 2 - 1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        try:
            res = minimize(error_func, params_init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success:
                params_opt = res.x
                U_opt = params_opt[:dim_u * R].reshape(dim_u, R)
                V_opt = params_opt[dim_u * R : dim_u * R + dim_v * R].reshape(dim_v, R)
                W_opt = params_opt[dim_u * R + dim_v * R :].reshape(dim_w, R)
                
                # Round to half-integers
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if is_valid(U_round, V_round, W_round):
                    best_rank = R
                    best_solution = {"U": U_round, "V": V_round, "W": W_round}
                    break # Found a valid lower rank, stop
        except Exception:
            pass
            
    # Convert to lists of lists as required
    U_list = best_solution["U"].tolist()
    V_list = best_solution["V"].tolist()
    W_list = best_solution["W"].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
