import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    size_u = n * m  # 12
    size_v = m * p  # 20
    size_w = p * n  # 15
    total_size = size_u + size_v + size_w  # 47
    
    def compute_error(U, V, W, R):
        """Compute sum of squared errors for all (i,j,k) triples."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    prod = np.multiply(np.multiply(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                    term = np.sum(prod)
                    diff = term - 1.0
                    error += diff * diff
        return error
    
    def error_func(params, R):
        U = params[:R * size_u].reshape(R, size_u).T  # (size_u, R)
        V = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v).T  # (size_v, R)
        W = params[R * (size_u + size_v) :].reshape(R, size_w).T  # (size_w, R)
        return compute_error(U, V, W, R)

    def round_and_check(U, V, W, R):
        """Round to nearest 0.5 and check exactness."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    prod = np.multiply(np.multiply(U_r[u_idx, :], V_r[v_idx, :]), W_r[w_idx, :])
                    term = np.sum(prod)
                    if abs(term - 1.0) > 1e-9:
                        return False, U_r, V_r, W_r
        return True, U_r, V_r, W_r

    def try_rank(R, max_attempts=5):
        """Try to find a valid decomposition of rank R."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values, but try different strategies
            if attempt == 0:
                # Small random integers/half-integers
                params = np.random.choice([-1, -0.5, 0, 0.5, 1], size=R * total_size)
            elif attempt == 1:
                # Uniform in [-1, 1]
                params = np.random.uniform(-1, 1, R * total_size)
            elif attempt == 2:
                # Sparse random
                params = np.random.uniform(-1, 1, R * total_size)
                params[np.random.choice(len(params), size=len(params)//2, replace=False)] = 0
            elif attempt == 3:
                # Larger range
                params = np.random.uniform(-2, 2, R * total_size)
            else:
                # Very small values
                params = np.random.uniform(-0.5, 0.5, R * total_size)
            
            # Optimize using L-BFGS-B
            try:
                res = minimize(error_func, params, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            except Exception:
                continue
            
            if res.fun < 1e-8:
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

    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Known results suggest rank around 30-35 for 3x4x5
    for R in range(20, 40):
        if time.time() > deadline:
            break
        result = try_rank(R, max_attempts=3)
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
