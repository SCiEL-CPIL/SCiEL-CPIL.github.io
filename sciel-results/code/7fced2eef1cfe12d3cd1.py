import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    n, m, p = 2, 3, 3
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Construct the target tensor T
    # T[u, v, w] = 1 if indices match, else 0
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for u in range(dim_u):
        i_u = u // m
        j_u = u % m
        for v in range(dim_v):
            j_v = v // p
            k_v = v % p
            if j_u != j_v:
                continue
            for w in range(dim_w):
                k_w = w // n
                i_w = w % n
                if k_v == k_w and i_w == i_u:
                    T[u, v, w] = 1.0
    
    # Function to compute the tensor from U, V, W matrices
    # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
    def tensor_from_factors(U, V, W):
        # Sum of outer products
        # T = sum_c U[:, c] * V[:, c] * W[:, c]
        # This can be done via einsum or loop
        # np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.einsum('uc,vc,wc->uvw', U, V, W)
    
    def objective(params, rank):
        if time.time() - start_time > time_limit:
            return float('inf')
        
        # Reshape params to U, V, W
        size_u = dim_u * rank
        size_v = dim_v * rank
        U = params[:size_u].reshape(dim_u, rank)
        V = params[size_u:size_u + size_v].reshape(dim_v, rank)
        W = params[size_u + size_v:].reshape(dim_w, rank)
        
        T_calc = tensor_from_factors(U, V, W)
        diff = T_calc - T
        return np.sum(diff * diff)
    
    def round_to_half(x):
        return np.round(2 * x) / 2.0
    
    best_solution = None
    best_rank = n * m * p
    best_obj = float('inf')
    
    # Try to find rank 15 decomposition
    target_rank = 15
    
    # If time permits, search for rank 15
    # We will use random restarts
    max_restarts = 500
    
    for r in range(max_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Initialize parameters
        # Use small random values
        np.random.seed(r) # Deterministic for reproducibility in this run, but different each restart
        params = np.random.uniform(-1, 1, size=(dim_u + dim_v + dim_w) * target_rank)
        
        # Optimize
        try:
            res = scipy.optimize.minimize(objective, params, args=(target_rank,), method='Nelder-Mead', 
                                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < best_obj:
                best_obj = res.fun
                # Check if it's close to zero
                if res.fun < 1e-8:
                    # Extract U, V, W
                    size_u = dim_u * target_rank
                    size_v = dim_v * target_rank
                    U = res.x[:size_u].reshape(dim_u, target_rank)
                    V = res.x[size_u:size_u + size_v].reshape(dim_v, target_rank)
                    W = res.x[size_u + size_v:].reshape(dim_w, target_rank)
                    
                    # Round to half-integers
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    
                    # Verify exactness
                    T_calc = tensor_from_factors(U_r, V_r, W_r)
                    if np.allclose(T_calc, T, atol=1e-9):
                        best_solution = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        break # Found exact rank 15, stop
                    else:
                        # Maybe rounding broke it, keep unrounded if it's the best so far?
                        # But the prompt requires integer/half-integer entries.
                        # So we must return rounded if valid, or find another solution.
                        pass
                        
        except Exception as e:
            continue
            
    # If rank 15 not found, try rank 16, 17... or just return schoolbook
    # Given time constraints, if 15 fails, we might not have time for others.
    # But schoolbook is always available.
    
    if best_solution is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_solution = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_solution

import scipy
