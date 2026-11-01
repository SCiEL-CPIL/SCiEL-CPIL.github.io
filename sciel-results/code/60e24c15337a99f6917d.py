import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 100  # seconds
    
    # Target: Rank 42 Strassen decomposition (Fallback)
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
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

    # Search for Rank 10 decomposition of 2x2x3
    def search_rank_10_2x2x3():
        if time.time() > deadline:
            return None
        
        # Tensor dimensions: U(4, R), V(6, R), W(6, R)
        # Target T[i, j, k] = 1 for i,j in 0..1, k in 0..2
        R = 10
        dim_U = 4
        dim_V = 6
        dim_W = 6
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            # Compute tensor product sum
            # T[i, j, k] = sum_s U[i,j,s] V[j,k,s] W[i,k,s]
            # Note: V index is j*3+k, W index is i*3+k
            T_calc = np.zeros((2, 2, 3))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T_calc[i, j, k] += U[i*2+j, s] * V[j*3+k, s] * W[i*3+k, s]
            
            T_target = np.ones((2, 2, 3))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        for _ in range(20): # 20 restarts
            if time.time() > deadline:
                break
            # Random initialization
            params = np.random.uniform(-2, 2, size=dim_U*R + dim_V*R + dim_W*R)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6:
            # Snap to half-integers
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify snapped solution
            T_calc = np.zeros((2, 2, 3))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            T_calc[i, j, k] += U_snapped[i*2+j, s] * V_snapped[j*3+k, s] * W_snapped[i*3+k, s]
            
            T_target = np.ones((2, 2, 3))
            if np.max(np.abs(T_calc - T_target)) < 1e-9:
                # Construct full 2x2x12 decomposition
                full_rank = R * 4
                full_U = [[0] * full_rank for _ in range(4)]
                full_V = [[0] * full_rank for _ in range(24)]
                full_W = [[0] * full_rank for _ in range(24)]
                
                for block in range(4):
                    offset = block * R
                    for s in range(R):
                        r = offset + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(U_snapped[i*2+j, s])
                        for j in range(2):
                            for k in range(3):
                                full_V[j*12 + (block*3 + k)][r] = float(V_snapped[j*3+k, s])
                        for i in range(2):
                            for k in range(3):
                                full_W[(block*3 + k)*2 + i][r] = float(W_snapped[i*3+k, s])
                                
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    # Search for Rank 13 decomposition of 2x2x4
    def search_rank_13_2x2x4():
        if time.time() > deadline:
            return None
            
        R = 13
        dim_U = 4
        dim_V = 8 # 2 * 4
        dim_W = 8 # 2 * 4
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((2, 2, 4))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            T_calc[i, j, k] += U[i*2+j, s] * V[j*4+k, s] * W[i*4+k, s]
            
            T_target = np.ones((2, 2, 4))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        for _ in range(10):
            if time.time() > deadline:
                break
            params = np.random.uniform(-2, 2, size=dim_U*R + dim_V*R + dim_W*R)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            T_calc = np.zeros((2, 2, 4))
            for s in range(R):
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            T_calc[i, j, k] += U_snapped[i*2+j, s] * V_snapped[j*4+k, s] * W_snapped[i*4+k, s]
            
            T_target = np.ones((2, 2, 4))
            if np.max(np.abs(T_calc - T_target)) < 1e-9:
                full_rank = R * 3
                full_U = [[0] * full_rank for _ in range(4)]
                full_V = [[0] * full_rank for _ in range(24)]
                full_W = [[0] * full_rank for _ in range(24)]
                
                for block in range(3):
                    offset = block * R
                    for s in range(R):
                        r = offset + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(U_snapped[i*2+j, s])
                        for j in range(2):
                            for k in range(4):
                                full_V[j*12 + (block*4 + k)][r] = float(V_snapped[j*4+k, s])
                        for i in range(2):
                            for k in range(4):
                                full_W[(block*4 + k)*2 + i][r] = float(W_snapped[i*4+k, s])
                                
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    # Try searches
    res = search_rank_10_2x2x3()
    if res:
        return res
        
    res = search_rank_13_2x2x4()
    if res:
        return res

    # Fallback
    return get_strassen_42()
