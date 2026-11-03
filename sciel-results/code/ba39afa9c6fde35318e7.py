import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

def _solve_block_decomposition(block_rank=12):
    n, m, p_block = 2, 3, 3
    r = block_rank
    
    # We want to decompose the all-ones tensor of size 2x3x3
    # T[i,j,k] = 1
    # into sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Note: The indices for W are k*n + i.
    # U indices: i*m + j
    # V indices: j*p_block + k
    
    for seed in range(10):
        np.random.seed(seed)
        
        U_init = np.random.randn(n*m, r) * 0.5
        V_init = np.random.randn(m*p_block, r) * 0.5
        W_init = np.random.randn(p_block*n, r) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p_block*r].reshape(m*p_block, r)
            W = x[n*m*r + m*p_block*r :].reshape(p_block*n, r)
            
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p_block):
                        val = np.sum(U[i*m+j] * V[j*p_block+k] * W[k*n+i])
                        errors.append(val - 1.0)
            return np.array(errors)
        
        try:
            result = opt.least_squares(residuals, x0, method='lm', max_nfev=2000)
            if np.linalg.norm(result.fun) < 1e-5:
                U = result.x[:n*m*r].reshape(n*m, r)
                V = result.x[n*m*r : n*m*r + m*p_block*r].reshape(m*p_block, r)
                W = result.x[n*m*r + m*p_block*r :].reshape(p_block*n, r)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p_block):
                            val = np.sum(U_rounded[i*m+j] * V_rounded[j*p_block+k] * W_rounded[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    return U_rounded, V_rounded, W_rounded
        except Exception:
            continue
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    block_decomp = _solve_block_decomposition(block_rank=12)
    
    if block_decomp is not None and time.time() < deadline:
        U_block, V_block, W_block = block_decomp
        r_block = 12
        total_rank = 3 * r_block
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for b in range(3):
            col_start = b * r_block
            col_end = (b + 1) * r_block
            
            U[:, col_start:col_end] = U_block
            
            for j in range(m):
                for k_block in range(3):
                    k_global = 3 * b + k_block
                    V[j*p + k_global, col_start:col_end] = V_block[j*3 + k_block]
            
            for k_block in range(3):
                k_global = 3 * b + k_block
                for i in range(n):
                    W[k_global*n + i, col_start:col_end] = W_block[k_block*n + i]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
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
