import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Target equations: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i in [0, n), j in [0, m), k in [0, p)
    
    def error_func(params, R):
        # params is a flat array of length R * (n*m + m*p + p*n)
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = size_u + size_v + size_w
        
        U = params[:R * size_u].reshape(R, size_u).T # (size_u, R)
        V = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
        W = params[R * (size_u + size_v) :].reshape(R, size_w).T
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Calculate sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # U is (size_u, R), so U[i*m+j, r] is U[i*m+j, r]
                    # V is (size_v, R), so V[j*p+k, r] is V[j*p+k, r]
                    # W is (size_w, R), so W[k*n+i, r] is W[k*n+i, r]
                    
                    # Efficient computation:
                    # term = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    term = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                    diff = term - 1.0
                    error += diff * diff
        return error

    def round_and_check(U, V, W, R):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term = np.dot(np.dot(U_r[i*m+j, :], V_r[j*p+k, :]), W_r[k*n+i, :])
                    if abs(term - 1.0) > 1e-9:
                        return False, U_r, V_r, W_r
        return True, U_r, V_r, W_r

    def try_rank(R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = size_u + size_v + size_w
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Use small integers or half-integers for initialization?
            # Random floats are better for optimization.
            params = np.random.uniform(-1, 1, R * total_size)
            
            # Optimize
            try:
                res = minimize(error_func, params, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.success and res.fun < 1e-8:
                # Extract U, V, W
                U = res.x[:R * size_u].reshape(R, size_u).T
                V = res.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T
                W = res.x[R * (size_u + size_v) :].reshape(R, size_w).T
                
                is_valid, U_r, V_r, W_r = round_and_check(U, V, W, R)
                if is_valid:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        return None

    # Try ranks from 34 to 43
    for R in range(34, 44):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result:
            return result

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
    return {"U": U, "V": V, "W": W}
