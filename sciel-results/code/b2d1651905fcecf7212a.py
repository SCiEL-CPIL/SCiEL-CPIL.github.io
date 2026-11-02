import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    target_rank = 14
    
    # Fallback Schoolbook
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    U_shape = (n*m, target_rank)
    V_shape = (m*p, target_rank)
    W_shape = (p*n, target_rank)
    
    def residuals(params):
        u_len = U_shape[0] * U_shape[1]
        v_len = V_shape[0] * V_shape[1]
        
        U = params[:u_len].reshape(U_shape)
        V = params[u_len:u_len+v_len].reshape(V_shape)
        W = params[u_len+v_len:].reshape(W_shape)
        
        T = np.zeros((n, m, p))
        for c in range(target_rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # Efficient update
            # T[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i*m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_col[j*p + k]
                        if v_val == 0: continue
                        w_val = w_col[k*n + i]
                        T[i, j, k] += u_val * v_val * w_val
                        
        return (T - 1).flatten()

    # Try to find rank 14 decomposition
    # We will try multiple random restarts.
    # To increase the chance of finding integer/half-integer solutions,
    # we can also try to initialize with integers.
    
    best_solution = None
    best_cost = np.inf

    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Using small integers or half-integers might help convergence to exact solutions
        # or at least get close.
        params = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], 
                                  size=(U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] + W_shape[0]*W_shape[1]))
        
        try:
            # Use Levenberg-Marquardt which is good for least squares
            result = least_squares(residuals, params, method='lm', max_nfev=1000)
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                
            if result.cost < 1e-8:
                # Check if it's close to half-integers
                u_len = U_shape[0] * U_shape[1]
                v_len = V_shape[0] * V_shape[1]
                
                U_sol = result.x[:u_len].reshape(U_shape)
                V_sol = result.x[u_len:u_len+v_len].reshape(V_shape)
                W_sol = result.x[u_len+v_len:].reshape(W_shape)
                
                # Round to nearest half-integer
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify the integer solution
                T_check = np.zeros((n, m, p))
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            u_val = U_int[i*m + j, c]
                            if u_val == 0: continue
                            for k in range(p):
                                v_val = V_int[j*p + k, c]
                                if v_val == 0: continue
                                w_val = W_int[k*n + i, c]
                                T_check[i, j, k] += u_val * v_val * w_val
                                
                if np.allclose(T_check, 1, atol=1e-9):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
        except Exception:
            continue

    # If we found a very good floating point solution but not exact integers,
    # we might still try to snap it, but if not, we return schoolbook.
    # The task requires exact decomposition. Schoolbook is exact.
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
