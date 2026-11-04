import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook solution as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_fb
    best_U, best_V, best_W = U_fb, V_fb, W_fb

    def error_func(params, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U].reshape(n, m, r)
        V_flat = params[size_U:size_U + size_V].reshape(m, p, r)
        W_flat = params[size_U + size_V:].reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_flat, V_flat, W_flat)
        
        err = np.sum((T - 1.0) ** 2)
        return err

    def check_and_snap(U, V, W, r):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        U_flat = U_s.reshape(n, m, r)
        V_flat = V_s.reshape(m, p, r)
        W_flat = W_s.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_flat, V_flat, W_flat)
        
        if np.allclose(T, 1.0, atol=1e-9):
            return U_s.flatten().tolist(), V_s.flatten().tolist(), W_s.flatten().tolist(), True
        return None, None, None, False

    for r in range(rank_fb - 1, 29, -1):
        if time.time() > deadline:
            break
            
        for restart in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + r * 100 + restart)
            params0 = np.random.randn(n * m * r + m * p * r + p * n * r) * 0.5
            
            try:
                res = opt.minimize(error_func, params0, args=(r,), method='Powell', 
                                   options={'maxiter': 500, 'ftol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                size_U = n * m * r
                size_V = m * p * r
                U_opt = res.x[:size_U].reshape(n * m, r)
                V_opt = res.x[size_U:size_U + size_V].reshape(m * p, r)
                W_opt = res.x[size_U + size_V:].reshape(p * n, r)
                
                U_list, V_list, W_list, valid = check_and_snap(U_opt, V_opt, W_opt, r)
                if valid:
                    best_rank = r
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break

    if best_rank == rank_fb:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
