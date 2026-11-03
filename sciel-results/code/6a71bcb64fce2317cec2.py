import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140.0

    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][c] = 1
                V_sb[j * p + k][c] = 1
                W_sb[k * n + i][c] = 1
                c += 1

    min_rank = max(n * m, m * p, p * n)
    
    # Target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # We can represent T as a list of non-zero indices
    non_zero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                non_zero.append((a, b, c_idx))
                
    def error_func(params, R):
        # params shape (n*m*R + m*p*R + p*n*R,)
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U].reshape(n * m, R)
        V_flat = params[size_U:size_U+size_V].reshape(m * p, R)
        W_flat = params[size_U+size_V:].reshape(p * n, R)
        
        err = 0.0
        for a, b, c_idx in non_zero:
            val = np.dot(U_flat[a], np.multiply(V_flat[b], W_flat[c_idx]))
            err += (val - 1.0) ** 2
        return err

    best_result = None
    best_rank = rank_sb
    best_params = None

    for R in range(min_rank, min_rank + 5):
        if time.time() - start_time > timeout:
            break
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # Try a few random starts
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
                
            x0 = np.random.randn(total_size) * 0.5
            
            res = minimize(error_func, x0, args=(R,), method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
            
            if res.success and res.fun < 1e-6:
                # Check if we can round to half-integers
                U_flat = res.x[:size_U].reshape(n * m, R)
                V_flat = res.x[size_U:size_U+size_V].reshape(m * p, R)
                W_flat = res.x[size_U+size_V:].reshape(p * n, R)
                
                # Round to nearest 0.5
                U_r = np.round(U_flat * 2) / 2
                V_r = np.round(V_flat * 2) / 2
                W_r = np.round(W_flat * 2) / 2
                
                # Verify exactness
                is_exact = True
                for a, b, c_idx in non_zero:
                    val = np.dot(U_r[a], np.multiply(V_r[b], W_r[c_idx]))
                    if not np.isclose(val, 1.0, atol=1e-9):
                        is_exact = False
                        break
                        
                if is_exact:
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
