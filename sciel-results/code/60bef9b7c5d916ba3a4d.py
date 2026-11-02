import numpy as np
import time
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-6):
        R = U.shape[1]
        err = 0.0
        V_mat = V.reshape(m, p, R) # V_mat[j, k, r]
        U_mat = U.reshape(n, m, R) # U_mat[i, j, r]
        W_mat = W.reshape(p, n, R) # W_mat[k, i, r]
        
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        # ... (existing code) ...
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    def search_lower_rank(start_rank, end_rank, max_iters=200):
        best_U, best_V, best_W = None, None, None
        best_rank = start_rank
        for R in range(start_rank, end_rank - 1, -1):
            if time.time() > deadline - 15:
                break
            # Try to find a solution of rank R using L-BFGS-B with multiple restarts
            found = False
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Random initialization with different scales
                scale = random.choice([0.1, 0.5, 1.0, 2.0])
                U = np.random.randn(n * m, R) * scale
                V = np.random.randn(m * p, R) * scale
                W = np.random.randn(p * n, R) * scale
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def objective_and_grad(params):
                    if time.time() > deadline - 15:
                        return float('inf'), np.zeros_like(params)
                    U_p = params[:n*m*R].reshape(n, m, R)
                    V_p = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                    W_p = params[n*m*R + m*p*R:].reshape(p, n, R)
                    
                    err = 0.0
                    grad_U = np.zeros_like(U_p)
                    grad_V = np.zeros_like(V_p)
                    grad_W = np.zeros_like(W_p)
                    
                    for i in range(n):
                        for k in range(p):
                            U_i = U_p[i]  # shape (m, R)
                            V_k = V_p[:, k]  # shape (m, R)
                            W_ki = W_p[k, i]  # shape (R,)
                            
                            # M_ik = U_i @ diag(W_ki) @ V_k.T
                            # = (U_i * W_ki) @ V_k.T
                            M_ik = (U_i * W_ki) @ V_k.T
                            diff = M_ik - np.eye(m)
                            err += np.sum(diff**2)
                            
                            # Gradient computation
                            # d(err)/d(M_ik) = 2 * diff
                            # M_ik = U_i @ diag(W_ki) @ V_k.T
                            # dM/dU_i = diag(W_ki) @ V_k.T
                            # dM/dV_k = U_i @ diag(W_ki)
                            # dM/dW_ki[r] = U_i[:, r] @ V_k[:, r].T (outer product)
                            
                            dM_dU = np.diag(W_ki) @ V_k.T  # shape (m, m)
                            dM_dV = U_i @ np.diag(W_ki)  # shape (m, m)
                            
                            grad_U_i = 2 * (diff @ dM_dU.T)  # shape (m, R)
                            grad_V_k = 2 * (dM_dV @ diff.T)  # shape (m, R)
                            
                            # For W: dM/dW_ki[r] = U_i[:, r] * V_k[:, r].T (outer product)
                            for r in range(R):
                                dM_dWr = np.outer(U_i[:, r], V_k[:, r])  # shape (m, m)
                                grad_W[k, i, r] += 2 * np.sum(diff * dM_dWr)
                    
                    return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                try:
                    res = minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', 
                                  options={'maxiter': max_iters, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.fun < 1e-6:
                        U_sol = res.x[:n*m*R].reshape(n, m, R).reshape(n*m, R)
                        V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m, p, R).reshape(m*p, R)
                        W_sol = res.x[n*m*R + m*p*R:].reshape(p, n, R).reshape(p*n, R)
                        # Verify
                        err = check_validity_residual(U_sol, V_sol, W_sol)
                        if err < 1e-6:
                            best_U, best_V, best_W = U_sol, V_sol, W_sol
                            best_rank = R
                            found = True
                            break # Found a solution for this rank, try lower
                except Exception:
                    pass
            if found:
                continue
            if time.time() > deadline - 15:
                break
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None, None, None, start_rank

    # First, get the known rank 55 solution
    U_55, V_55, W_55 = get_rank_55_solution()
    err_55 = check_validity_residual(U_55, V_55, W_55)
    
    if err_55 > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank
    U_best, V_best, W_best, rank_best = search_lower_rank(54, 40, max_iters=200)
    
    if U_best is not None:
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    else:
        return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
