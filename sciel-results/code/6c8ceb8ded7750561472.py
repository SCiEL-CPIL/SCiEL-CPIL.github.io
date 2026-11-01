import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from scipy.optimize import minimize, least_squares

def construct():
    n, m, p = 2, 2, 5
    
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    def residual_full(params, rank):
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[k*p + l, c] * W[l*n + i, c]
                        res.append(val - target)
        return np.array(res)

    def check_exact_full(U, V, W, rank, tol=1e-5):
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        target = 1.0 if j == k else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j, c] * V[k*p + l, c] * W[l*n + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def try_find_full_decomposition(target_rank, num_restarts=20):
        if time.time() > DEADLINE - 10:
            return None
        
        size = n*m*target_rank + m*p*target_rank + p*n*target_rank
        best_res = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 10:
                break
            np.random.seed(restart * 101 + target_rank)
            x0 = np.random.uniform(-1, 1, size=size)
            
            # Use least_squares for better stability
            try:
                res = least_squares(residual_full, x0, args=(target_rank,), method='lm', 
                                    max_nfev=1000)
                err = np.sum(res.fun**2)
                if err < best_err:
                    best_err = err
                    best_res = res
                if err < 1e-6:
                    break
            except Exception:
                continue
                
        if best_res and best_err < 1e-5:
            U = best_res.x[:n*m*target_rank].reshape((n*m, target_rank))
            V = best_res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
            W = best_res.x[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
            
            if check_exact_full(U, V, W, target_rank):
                # Snap to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_exact_full(U_r, V_r, W_r, target_rank):
                    return (U_r, V_r, W_r)
        return None

    # Try rank 15
    if time.time() < DEADLINE - 20:
        sol = try_find_full_decomposition(15, num_restarts=30)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 16
    if time.time() < DEADLINE - 20:
        sol = try_find_full_decomposition(16, num_restarts=50)
        if sol:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    return get_rank_18_solution()
