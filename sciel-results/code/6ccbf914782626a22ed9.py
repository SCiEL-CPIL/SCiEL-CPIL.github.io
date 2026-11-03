import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140

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

    def check_decomp(U, V, W, atol=1e-6):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T, atol=atol)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def snap_to_quarter(U, V, W):
        U_s = np.round(U * 4) / 4
        V_s = np.round(V * 4) / 4
        W_s = np.round(W * 4) / 4
        return U_s, V_s, W_s

    def try_rank_powell(rank, num_trials=5):
        """Try to find a rank-r decomposition using Powell optimization."""
        size_u, size_v, size_w = n*m, m*p, p*n
        total_vars = (size_u + size_v + size_w) * rank
        
        def objective(x):
            U = x[:size_u*rank].reshape(size_u, rank)
            V = x[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
            W = x[size_u*rank + size_v*rank:].reshape(size_w, rank)
            T_decomp = compute_decomp(U, V, W)
            return np.sum((T - T_decomp) ** 2)

        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(trial * 1000 + rank)
            # Initialize with small random values
            x0 = np.random.randn(total_vars) * 0.5
            
            try:
                # Powell is derivative-free and robust
                res = minimize(objective, x0, method='Powell', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12})
                
                if res.fun < 1e-8:
                    x_opt = res.x
                    U = x_opt[:size_u*rank].reshape(size_u, rank)
                    V = x_opt[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
                    W = x_opt[size_u*rank + size_v*rank:].reshape(size_w, rank)
                    
                    # Try to snap to nice numbers
                    U_s, V_s, W_s = snap_to_half(U, V, W)
                    if check_decomp(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    
                    U_s2, V_s2, W_s2 = snap_to_quarter(U, V, W)
                    if check_decomp(U_s2, V_s2, W_s2):
                        return U_s2, V_s2, W_s2
                    
                    if check_decomp(U, V, W):
                        return U, V, W
                        
            except Exception:
                pass
        
        return None

    # Try ranks from low to high
    # We know 36 is possible. Let's try to beat it.
    # 3x3x3 is 23. 3x3x4 is likely around 28-30.
    for rank in range(28, 36):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
        
        result = try_rank_powell(rank, num_trials=10)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = rank
            # If we find a very low rank, we can stop early if we are confident
            if rank <= 28:
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
