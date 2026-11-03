import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def compute_decomp(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W, optimize=True)

    def check_decomp(U, V, W):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T, atol=1e-9)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def objective(params, rank):
        size_u, size_v, size_w = n*m, m*p, p*n
        U = params[:size_u*rank].reshape(size_u, rank)
        V = params[size_u*rank:size_u*rank+size_v*rank].reshape(size_v, rank)
        W = params[size_u*rank+size_v*rank:].reshape(size_w, rank)
        T_decomp = compute_decomp(U, V, W)
        return np.sum((T - T_decomp) ** 2)

    def try_rank(rank, num_trials=10):
        """Try to find a rank-r decomposition using L-BFGS-B."""
        size_u, size_v, size_w = n*m, m*p, p*n
        total_vars = size_u * rank + size_v * rank + size_w * rank
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            # Initialize with small random values
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(size_u, rank) * 0.5
            V = np.random.randn(size_v, rank) * 0.5
            W = np.random.randn(size_w, rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Run L-BFGS-B
            res = minimize(objective, params, args=(rank,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-8})
            
            if res.success and res.fun < 1e-6:
                U_opt = res.x[:size_u*rank].reshape(size_u, rank)
                V_opt = res.x[size_u*rank:size_u*rank+size_v*rank].reshape(size_v, rank)
                W_opt = res.x[size_u*rank+size_v*rank:].reshape(size_w, rank)
                
                # Polish with ALS
                # ... (Optional, but L-BFGS is usually good enough)
                
                # Check if it works
                if check_decomp(U_opt, V_opt, W_opt):
                    # Try to snap to half-integers
                    U_s, V_s, W_s = snap_to_half(U_opt, V_opt, W_opt)
                    if check_decomp(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    else:
                        # Return the floating point result if it's exact
                        return U_opt, V_opt, W_opt
        
        return None

    # Try ranks from 24 to 35
    for rank in range(24, 36):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
            
        result = try_rank(rank, num_trials=10)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = rank
            # If we found a low rank, we can stop
            if rank <= 25:
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
