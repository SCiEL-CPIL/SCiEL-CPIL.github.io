import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    total_time_limit = 110.0
    start_time = time.time()
    
    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
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
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    U, V, W = get_rank_55_solution()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 55

    def check_validity(U, V, W, tol=1e-6):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        err = 0.0
        for i in range(n):
            for k in range(p):
                W_k_i = W_mat[k, i, :]
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                A = U_i * W_k_i[None, :]
                M_ik = A.T @ V_k
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err < tol

    def tensor_error(params, R):
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R :].reshape(p*n, R)
        
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                W_k_i = W_mat[k, i, :]
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                A = U_i * W_k_i[None, :]
                M_ik = A.T @ V_k
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    # Try to find rank 54 by dropping one column and adjusting
    def try_rank_54():
        if time.time() - start_time > total_time_limit - 20:
            return None, None, None
        
        best_err = float('inf')
        best_params = None
        
        for drop_col in range(55):
            if time.time() - start_time > total_time_limit - 20:
                break
            
            U_init = np.delete(U, drop_col, axis=1)
            V_init = np.delete(V, drop_col, axis=1)
            W_init = np.delete(W, drop_col, axis=1)
            
            # Add small noise for local search
            noise_scale = 0.05
            U_init = U_init + np.random.normal(0, noise_scale, U_init.shape)
            V_init = V_init + np.random.normal(0, noise_scale, V_init.shape)
            W_init = W_init + np.random.normal(0, noise_scale, W_init.shape)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Simple gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            v = np.zeros_like(params_init)
            
            for step in range(200):
                if time.time() - start_time > total_time_limit - 15:
                    break
                
                params = params_init
                U_curr = params[:n*m*54].reshape(n*m, 54)
                V_curr = params[n*m*54 : n*m*54 + m*p*54].reshape(m*p, 54)
                W_curr = params[n*m*54 + m*p*54 :].reshape(p*n, 54)
                
                # Compute gradient numerically
                grad = np.zeros_like(params)
                for idx in range(0, len(params), 50):
                    end = min(idx + 50, len(params))
                    for j in range(idx, end):
                        eps = 1e-5
                        params_plus = params.copy()
                        params_plus[j] += eps
                        err_plus = tensor_error(params_plus, 54)
                        
                        params_minus = params.copy()
                        params_minus[j] -= eps
                        err_minus = tensor_error(params_minus, 54)
                        
                        grad[j] = (err_plus - err_minus) / (2 * eps)
                
                v = momentum * v - lr * grad
                params_init = params_init + v
                
                if np.linalg.norm(grad) < 1e-8:
                    break
            
            err_final = tensor_error(params_init, 54)
            if err_final < best_err:
                best_err = err_final
                best_params = params_init.copy()
            
            if best_err < 1e-4:
                break
        
        if best_params is not None and best_err < 1e-4:
            U_new = best_params[:n*m*54].reshape(n*m, 54)
            V_new = best_params[n*m*54 : n*m*54 + m*p*54].reshape(m*p, 54)
            W_new = best_params[n*m*54 + m*p*54 :].reshape(p*n, 54)
            
            # Snap to nearest 0.5
            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2
            
            if check_validity(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        return None, None, None

    U54, V54, W54 = try_rank_54()
    if U54 is not None:
        best_U, best_V, best_W = U54, V54, W54
        best_rank = 54

    # Try structured rank 37 search
    # The idea: 5 blocks of 2x2 (each can use rank 7 Strassen) + 5 blocks of 2x1 (each rank 1)
    # Total: 5*7 + 5*1 = 40, but we can do better with shared structure
    def try_rank_37():
        if time.time() - start_time > total_time_limit - 10:
            return None, None, None
        
        # Initialize with structured pattern
        R = 37
        U_init = np.zeros((n*m, R))
        V_init = np.zeros((m*p, R))
        W_init = np.zeros((p*n, R))
        
        # Part A: j=0, 10 terms
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U_init[u_idx, k] = 1
                V_init[v_idx, k] = 1
                W_init[w_idx, k] = 1
        
        # Part B: 27 terms for the 2x2 blocks
        col = 10
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
            
            # Use 5 Strassen-like terms per block (reduced from 7)
            # M1 = (a11 + a22)(b11 + b22)
            if col < R:
                U_init[u_11, col] += 1; U_init[u_22, col] += 1
                V_init[v_11, col] += 1; V_init[v_22, col] += 1
                W_init[w_00, col] += 1; W_init[w_11, col] += 1
                col += 1
            
            # M2 = (a21 + a22)b11
            if col < R:
                U_init[u_21, col] += 1; U_init[u_22, col] += 1
                V_init[v_11, col] += 1
                W_init[w_10, col] += 1; W_init[w_11, col] += -1
                col += 1
            
            # M3 = a11(b12 - b22)
            if col < R:
                U_init[u_11, col] += 1
                V_init[v_12, col] += 1; V_init[v_22, col] += -1
                W_init[w_01, col] += 1; W_init[w_11, col] += 1
                col += 1
            
            # M4 = a22(b21 - b11)
            if col < R:
                U_init[u_22, col] += 1
                V_init[v_21, col] += 1; V_init[v_11, col] += -1
                W_init[w_00, col] += 1; W_init[w_10, col] += 1
                col += 1
            
            # M5 = (a11 + a12)b22
            if col < R:
                U_init[u_11, col] += 1; U_init[u_12, col] += 1
                V_init[v_22, col] += 1
                W_init[w_00, col] += -1; W_init[w_01, col] += 1
                col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            if col < R:
                U_init[u_21, col] += 1; U_init[u_11, col] += -1
                V_init[v_11, col] += 1; V_init[v_12, col] += 1
                W_init[w_11, col] += 1
                col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            if col < R:
                U_init[u_12, col] += 1; U_init[u_22, col] += -1
                V_init[v_21, col] += 1; V_init[v_22, col] += 1
                W_init[w_00, col] += 1
                col += 1
        
        if col > R:
            # Truncate to R columns
            U_init = U_init[:, :R]
            V_init = V_init[:, :R]
            W_init = W_init[:, :R]
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Gradient descent
        lr = 0.005
        momentum = 0.9
        v = np.zeros_like(params_init)
        
        for step in range(300):
            if time.time() - start_time > total_time_limit - 5:
                break
            
            # Compute gradient
            grad = np.zeros_like(params_init)
            for idx in range(0, len(params), 30):
                end = min(idx + 30, len(params))
                for j in range(idx, end):
                    eps = 1e-5
                    params_plus = params_init.copy()
                    params_plus[j] += eps
                    err_plus = tensor_error(params_plus, R)
                    
                    params_minus = params_init.copy()
                    params_minus[j] -= eps
                    err_minus = tensor_error(params_minus, R)
                    
                    grad[j] = (err_plus - err_minus) / (2 * eps)
            
            v = momentum * v - lr * grad
            params_init = params_init + v
            
            if np.linalg.norm(grad) < 1e-8:
                break
        
        err_final = tensor_error(params_init, R)
        if err_final < 1e-4:
            U_new = params_init[:n*m*R].reshape(n*m, R)
            V_new = params_init[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_new = params_init[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Snap to nearest 0.5
            U_new = np.round(U_new * 2) / 2
            V_new = np.round(V_new * 2) / 2
            W_new = np.round(W_new * 2) / 2
            
            if check_validity(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        return None, None, None

    U37, V37, W37 = try_rank_37()
    if U37 is not None:
        best_U, best_V, best_W = U37, V37, W37
        best_rank = 37

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
