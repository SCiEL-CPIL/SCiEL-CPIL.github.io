import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    # Fallback: Rank 42 Strassen decomposition (6 blocks of 2x2x2)
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard Strassen coefficients for 2x2x2
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * m + j_loc][r] = val
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * p + (2 * b + k_loc)][r] = val
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * n + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        T_calc = np.zeros((n, m, p))
        for s in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
        T_target = np.ones((n, m, p))
        max_err = np.max(np.abs(T_calc - T_target))
        return max_err < 1e-9

    def search_rank(R, n, m, p, num_restarts=20, maxiter=500):
        if time.time() > deadline:
            return None
            
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n, m, p))
            for s in range(R):
                u_col = U[:, s].reshape(n, m)
                v_col = V[:, s].reshape(m, p)
                w_col = W[:, s].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        np.random.seed(42)
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            params = np.random.uniform(-1, 1, size=dim_U*R + dim_V*R + dim_W*R)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': 2000, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                return U_snapped, V_snapped, W_snapped
        return None

    # Try to find rank 12 for 2x2x4 (Total rank 36)
    res = search_rank(12, 2, 2, 4, num_restarts=20, maxiter=300)
    if res:
        U_sub, V_sub, W_sub = res
        full_rank = 36
        full_U = [[0] * full_rank for _ in range(4)]
        full_V = [[0] * full_rank for _ in range(24)]
        full_W = [[0] * full_rank for _ in range(24)]
        
        for block in range(3):
            offset = block * 12
            for s in range(12):
                r = offset + s
                for i in range(2):
                    for j in range(2):
                        full_U[i*2+j][r] = float(U_sub[i*2+j, s])
                for j in range(2):
                    for k in range(4):
                        full_V[j*12 + (block*4 + k)][r] = float(V_sub[j*4+k, s])
                for i in range(2):
                    for k in range(4):
                        full_W[(block*4 + k)*2 + i][r] = float(W_sub[i*4+k, s])
                        
        return {"U": full_U, "V": full_V, "W": full_W}

    # Try to find rank 10 for 2x2x3 (Total rank 40)
    res = search_rank(10, 2, 2, 3, num_restarts=20, maxiter=300)
    if res:
        U_sub, V_sub, W_sub = res
        full_rank = 40
        full_U = [[0] * full_rank for _ in range(4)]
        full_V = [[0] * full_rank for _ in range(24)]
        full_W = [[0] * full_rank for _ in range(24)]
        
        for block in range(4):
            offset = block * 10
            for s in range(10):
                r = offset + s
                for i in range(2):
                    for j in range(2):
                        full_U[i*2+j][r] = float(U_sub[i*2+j, s])
                for j in range(2):
                    for k in range(3):
                        full_V[j*12 + (block*3 + k)][r] = float(V_sub[j*3+k, s])
                for i in range(2):
                    for k in range(3):
                        full_W[(block*3 + k)*2 + i][r] = float(W_sub[i*3+k, s])
                        
        return {"U": full_U, "V": full_V, "W": full_W}

    # Fallback
    return get_strassen_42()
