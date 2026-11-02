"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _find_m223_rank10(timeout=10):
    """
    Attempts to find a rank-10 decomposition for M_{2,2,3} with half-integer entries.
    """
    n, m, p = 2, 2, 3
    r = 10
    
    def unpack(params):
        u = params[0:4*r].reshape(n*m, r)
        v = params[4*r:4*r+6*r].reshape(m*p, r)
        w = params[4*r+6*r:4*r+12*r].reshape(p*n, r)
        return u, v, w
    
    def objective(params):
        u, v, w = unpack(params)
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += u[i*m+j, rr] * v[j*p+k, rr] * w[k*n+i, rr]
                    residuals.append(val - 1.0)
        return np.sum(np.array(residuals)**2)
    
    def half_integer_penalty(params):
        # Penalty for being far from half-integers
        u, v, w = unpack(params)
        all_vals = np.concatenate([u.flatten(), v.flatten(), w.flatten()])
        # Distance to nearest 0.5
        dists = np.abs(all_vals * 2 - np.round(all_vals * 2)) / 2
        return np.sum(dists**2)

    def combined_objective(params):
        return objective(params) + 0.01 * half_integer_penalty(params)

    best_sol = None
    start_time = time.time()
    
    # Try a few random seeds
    for seed in range(5):
        if time.time() - start_time > timeout:
            break
        np.random.seed(100 + seed)
        x0 = np.random.randn(12*r) * 0.1
        # Initialize with some structure? 
        # Maybe start from a known rank 11 solution and project?
        # Rank 11 solution: 7 Strassen + 4 Schoolbook for last column?
        # No, M_{2,2,3} is not M_{2,2,2} + M_{2,2,1}.
        # It is M_{2,2,2} (k=0,1) + M_{2,2,1} (k=2).
        # So we can initialize U, V, W with the rank 11 solution and set the last 11th rank to 0?
        # This might help.
        
        # Construct rank 11 solution for M_{2,2,3}
        # U11: 4x11, V11: 6x11, W11: 6x11
        U11 = np.zeros((4, 11))
        V11 = np.zeros((6, 11))
        W11 = np.zeros((6, 11))
        
        # Strassen for k=0,1
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
        # Schoolbook for k=2
        U_school = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_school = [
            [1, 0, 1, 0], # j=0, k=2
            [0, 1, 0, 1]  # j=1, k=2
        ]
        W_school = [
            [1, 1, 0, 0], # i=0, k=2
            [0, 0, 1, 1]  # i=1, k=2
        ]
        
        for i in range(4):
            for rr in range(7):
                U11[i, rr] = U_strassen[i][rr]
            for rr in range(4):
                U11[i, 7+rr] = U_school[i][rr]
                
        for j in range(2):
            for local_k in range(2):
                for rr in range(7):
                    V11[j*2+local_k, rr] = V_strassen[j*2+local_k][rr]
            for rr in range(4):
                V11[j*3+2, 7+rr] = V_school[j][rr]
                
        for local_k in range(2):
            for i in range(2):
                for rr in range(7):
                    W11[local_k*2+i, rr] = W_strassen[local_k*2+i][rr]
            for i in range(2):
                for rr in range(4):
                    W11[2*2+i, 7+rr] = W_school[i][rr]
                    
        # Initialize x0 with U11, V11, W11 but only first 10 columns?
        # Or random?
        # Let's try random first, it's more robust for finding global minima in non-convex problems.
        
        try:
            res = minimize(combined_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'disp': False})
            
            if res.fun < 1e-6:
                u, v, w = unpack(res.x)
                u_rounded = np.round(u * 2) / 2
                v_rounded = np.round(v * 2) / 2
                w_rounded = np.round(w * 2) / 2
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(r):
                                val += u_rounded[i*m+j, rr] * v_rounded[j*p+k, rr] * w_rounded[k*n+i, rr]
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
