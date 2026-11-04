import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 4, 7
    
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    def verify(U, V, W, tol=1e-5):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        val = np.dot(W[w_idx, :], U[u_idx, :] * V[v_idx, :])
                        target = 1.0 if j == j_prime else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err
        return max_err < tol

    def build_residuals(U, V, W):
        r = U.shape[1]
        res = np.zeros(n * p * m * m)
        idx = 0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        res[idx] = np.dot(W[w_idx, :], U[u_idx, :] * V[v_idx, :]) - (1.0 if j == j_prime else 0.0)
                        idx += 1
        return res

    def residual(params, r):
        U = params[0:r*dim_U].reshape(r, dim_U).T
        V = params[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
        W = params[r*(dim_U+dim_V):].reshape(r, dim_W).T
        return build_residuals(U, V, W)

    # Schoolbook decomposition
    def schoolbook():
        rank_school = n * m * p
        U = np.zeros((dim_U, rank_school))
        V = np.zeros((dim_V, rank_school))
        W = np.zeros((dim_W, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Use a structured initialization based on polynomial interpolation
    # For 3x4x7, try ranks 45, 42, 40, 38, 36, 34, 32, 30
    
    for r in [45, 42, 40, 38, 36, 34, 32, 30]:
        if time.time() > deadline - 30:
            break
        
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(2000 + r * 10 + restart)
            
            # Structured initialization: each term corresponds to a (i,j,k) with random weights
            U_init = np.zeros((dim_U, r))
            V_init = np.zeros((dim_V, r))
            W_init = np.zeros((dim_W, r))
            
            # Create r random (i,j,k) triples
            for col in range(r):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                # Use random weights to allow flexibility
                w_u = np.random.uniform(0.5, 1.5)
                w_v = np.random.uniform(0.5, 1.5)
                w_w = np.random.uniform(0.5, 1.5)
                U_init[i * m + j, col] = w_u
                V_init[j * p + k, col] = w_v
                W_init[k * n + i, col] = w_w
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual, params_init, args=(r,), method='trf',
                                   max_nfev=300, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                
                max_err = np.max(np.abs(sol.fun))
                if max_err < 1e-2:
                    U_sol = sol.x[0:r*dim_U].reshape(r, dim_U).T
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(r, dim_W).T
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-2):
                        # Polish
                        params_polish = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                        sol_polish = least_squares(residual, params_polish, args=(r,), method='trf',
                                                   max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                        
                        U_final = sol_polish.x[0:r*dim_U].reshape(r, dim_U).T
                        V_final = sol_polish.x[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
                        W_final = sol_polish.x[r*(dim_U+dim_V):].reshape(r, dim_W).T
                        
                        if verify(U_final, V_final, W_final, tol=1e-4):
                            best_result = (U_final, V_final, W_final)
                            best_rank = r
                            break
            except Exception:
                continue
        
        if best_rank < r:
            break

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
