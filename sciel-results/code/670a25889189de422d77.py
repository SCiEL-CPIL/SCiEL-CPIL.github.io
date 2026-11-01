import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target tensor
    # U: n*m, V: m*p, W: p*n
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    target_flat = target.flatten()
    
    best_solution = None
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(dim_u)]
    best_V = [[0] * best_rank for _ in range(dim_v)]
    best_W = [[0] * best_rank for _ in range(dim_w)]
    
    # Initialize best with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1

    # Try ranks from 24 upwards
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        max_restarts = 10 if R < 28 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randint(-2, 3, size=(dim_u, R))
            V = np.random.randint(-2, 3, size=(dim_v, R))
            W = np.random.randint(-2, 3, size=(dim_w, R))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residual(params):
                U_p = params[:dim_u*R].reshape(dim_u, R)
                V_p = params[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                W_p = params[dim_u*R+dim_v*R:].reshape(dim_w, R)
                
                T_prime = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                return (T_prime.flatten() - target_flat)
            
            # Use scipy optimize
            try:
                from scipy.optimize import least_squares
                result = least_squares(residual, params, method='lm', max_nfev=1000)
                res_norm = result.cost * 2  # cost is 0.5 * sum(res^2)
                
                if res_norm < 1e-6:
                    # Check for exactness with rounding
                    U_final = result.x[:dim_u*R].reshape(dim_u, R)
                    V_final = result.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                    W_final = result.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_final * 2) / 2.0
                    V_rounded = np.round(V_final * 2) / 2.0
                    W_rounded = np.round(W_final * 2) / 2.0
                    
                    T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, target, atol=1e-9):
                        # Convert to list of lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass

    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
