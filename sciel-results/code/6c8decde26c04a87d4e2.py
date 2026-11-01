import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 20
    target_rank = 20
    start_time = time.time()
    time_limit = 10.0 # 10 seconds for search
    
    # Construct target tensor T (sparse)
    # T[a, b, c] = 1 if indices match, else 0
    # a: i*m + j
    # b: j*p + k
    # c: k*n + i
    # We only need to check the 24 non-zero entries
    # But for optimization, we can use the full tensor or just the non-zero ones
    # Let's use the non-zero ones to save memory/time in residual calculation
    
    # List of (a, b, c) indices where T=1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
                
    num_nonzero = len(indices)
    
    # Parameters: U (4x20), V (12x20), W (12x20)
    # Flatten for optimization
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    def residual(x):
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u+size_v]
        W_flat = x[size_u+size_v:]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        res = np.zeros(num_nonzero)
        for idx, (a, b, c) in enumerate(indices):
            # Sum_r U[a,r] * V[b,r] * W[c,r]
            res[idx] = np.sum(U[a, :] * V[b, :] * W[c, :])
        
        # Target is 1 for all these indices
        return res - 1.0
        
    # Initial guess: Random small values
    # We try a few random restarts
    best_result = None
    np.random.seed(42)
    
    # We use the "Flattening and SVD" idea to initialize?
    # Actually, random init is often better for non-convex tensor decomposition.
    # But we can try initializing with the Rank 21 solution truncated?
    # No, dimensions don't match.
    
    # Let's try 2 random restarts with limited iterations.
    for attempt in range(2):
        if time.time() - start_time > time_limit - 2.0:
            break
            
        x0 = np.random.normal(0, 0.5, total_size)
        
        try:
            # Use least_squares
            # 'lm' is fast but doesn't support bounds. 'trf' is robust.
            # We don't have bounds, but we want integer-like solutions eventually.
            # Least squares finds real solutions.
            res = opt.least_squares(residual, x0, method='trf', max_nfev=200, xtol=1e-8, ftol=1e-8)
            
            if time.time() - start_time > time_limit - 1.0:
                break
                
            if res.success or np.linalg.norm(res.fun) < 1e-5:
                # Check if we can snap to integers/half-integers
                U_opt = res.x[:size_u].reshape((n*m, target_rank))
                V_opt = res.x[size_u:size_u+size_v].reshape((m*p, target_rank))
                W_opt = res.x[size_u+size_v:].reshape((p*n, target_rank))
                
                # Snap to nearest 0.5
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify exactness with snapped values
                max_err = 0
                for a, b, c in indices:
                    val = np.sum(U_int[a, :] * V_int[b, :] * W_int[c, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                
                if max_err < 1e-9:
                    # Convert to list of lists
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    # Clean up -0.0
                    for i in range(len(U_list)):
                        for j in range(len(U_list[i])):
                            if U_list[i][j] == 0: U_list[i][j] = 0
                    for i in range(len(V_list)):
                        for j in range(len(V_list[i])):
                            if V_list[i][j] == 0: V_list[i][j] = 0
                    for i in range(len(W_list)):
                        for j in range(len(W_list[i])):
                            if W_list[i][j] == 0: W_list[i][j] = 0
                            
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break # Found a valid rank 20, stop searching
        
        except Exception:
            pass
            
    if best_result:
        return best_result
        
    return get_rank_21()
