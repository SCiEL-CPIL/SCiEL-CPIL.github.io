"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _find_m223_rank10(timeout=15):
    """
    Attempts to find a rank-10 decomposition for M_{2,2,3} with half-integer entries.
    """
    n, m, p = 2, 2, 3
    r = 10
    dim_u = n * m # 4
    dim_v = m * p # 6
    dim_w = p * n # 6
    total_vars = dim_u * r + dim_v * r + dim_w * r # 160
    
    def unpack(params):
        u = params[0:dim_u*r].reshape(dim_u, r)
        v = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
        w = params[dim_u*r+dim_v*r:total_vars].reshape(dim_w, r)
        return u, v, w
    
    def objective(params):
        u, v, w = unpack(params)
        residuals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residuals[idx] = np.sum(u[i*m+j] * v[j*p+k] * w[k*n+i]) - 1.0
                    idx += 1
        return np.sum(residuals**2)

    def half_integer_penalty(params):
        u, v, w = unpack(params)
        all_vals = np.concatenate([u.flatten(), v.flatten(), w.flatten()])
        dists = np.abs(all_vals * 2 - np.round(all_vals * 2)) / 2
        return np.sum(dists**2)

    def combined_objective(params):
        return objective(params) + 0.01 * half_integer_penalty(params)

    start_time = time.time()
    
    # Construct rank 11 solution for M_{2,2,3} to use as initialization
    U11 = np.zeros((dim_u, 11))
    V11 = np.zeros((dim_v, 11))
    W11 = np.zeros((dim_w, 11))
    
    # Strassen for k=0,1
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    # Schoolbook for k=2
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0], # j=0, k=2
        [0, 1, 0, 1]  # j=1, k=2
    ])
    W_school = np.array([
        [1, 1, 0, 0], # i=0, k=2
        [0, 0, 1, 1]  # i=1, k=2
    ])
    
    for i in range(4):
        U11[i, :7] = U_strassen[i]
        U11[i, 7:] = U_school[i]
        
    for j in range(2):
        for local_k in range(2):
            V11[j*2+local_k, :7] = V_strassen[j*2+local_k]
        V11[j*3+2, 7:] = V_school[j]
        
    for local_k in range(2):
        for i in range(2):
            W11[local_k*2+i, :7] = W_strassen[local_k*2+i]
        for i in range(2):
            W11[2*2+i, 7:] = W_school[i]
            
    strategies = []
    for seed in range(5):
        strategies.append(('random', seed))
    for zero_col in range(11):
        strategies.append(('project', zero_col))
    for seed in range(5):
        strategies.append(('perturb', seed))

    for strat_type, strat_param in strategies:
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(1000 + strat_param)
        
        if strat_type == 'random':
            x0 = np.random.randn(total_vars) * 0.5
        elif strat_type == 'project':
            U_init = U11[:, :10]
            V_init = V11[:, :10]
            W_init = W11[:, :10]
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            x0 += np.random.randn(total_vars) * 0.01
        elif strat_type == 'perturb':
            U_init = U11[:, :10]
            V_init = V11[:, :10]
            W_init = W11[:, :10]
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            col_to_perturb = strat_param % 10
            x0[dim_u*r + col_to_perturb] += np.random.randn() * 0.1
            x0[dim_u*r + dim_v*r + col_to_perturb] += np.random.randn() * 0.1
            
        try:
            res = minimize(combined_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res.fun < 1e-5:
                u, v, w = unpack(res.x)
                u_rounded = np.round(u * 2) / 2
                v_rounded = np.round(v * 2) / 2
                w_rounded = np.round(w * 2) / 2
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(u_rounded[i*m+j] * v_rounded[j*p+k] * w_rounded[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    return u_rounded, v_rounded, w_rounded
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Rank 53 solution
    rank_fallback = 53
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U_fb[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V_fb[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V_fb[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_fb[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U_fb[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V_fb[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W_fb[k_last * n + i][c] = W_school[i][r]
            
    # Try to find rank 52 solution
    # 6 blocks of 2x2x2 (rank 7 each) = 42
    # 1 block of 2x2x3 (rank 10) = 10
    # Total 52
    if time.time() < deadline - 20:
        sol = _find_m223_rank10(timeout=15)
        if sol is not None:
            u_m223, v_m223, w_m223 = sol
            rank_new = 52
            U_new = [[0] * rank_new for _ in range(n * m)]
            V_new = [[0] * rank_new for _ in range(m * p)]
            W_new = [[0] * rank_new for _ in range(p * n)]
            
            col_offset = 0
            for block in range(6):
                k0 = 2 * block
                k1 = 2 * block + 1
                for r in range(7):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = U_strassen[i * 2 + j][r]
                    for j in range(2):
                        V_new[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                        V_new[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                    for k in range(2):
                        kk = k0 + k
                        for i in range(2):
                            W_new[kk * n + i][c] = W_strassen[k * 2 + i][r]
                col_offset += 7
            
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U_new[i * m + j][c] = float(u_m223[i * 2 + j, r])
                for j in range(2):
                    for local_k in range(3):
                        global_k = 12 + local_k
                        V_new[j * p + global_k][c] = float(v_m223[j * 3 + local_k, r])
                for local_k in range(3):
                    global_k = 12 + local_k
                    for i in range(2):
                        W_new[global_k * n + i][c] = float(w_m223[local_k * 2 + i, r])
            
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
