"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def compute_residuals(U, V, W, rank):
        residuals = []
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = V[k*p:(k+1)*p, :]
                w_val = W[k*n + i]
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                target = np.eye(m)
                residuals.append(mat.flatten() - target.flatten())
        return np.concatenate(residuals)
    
    def check_exact(U, V, W, rank):
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-5):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-5):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-5):
            return False
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_slice = V_int[k*p:(k+1)*p, :]
                w_val = W_int[k*n + i]
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-6):
                    return False
        return True
    
    min_rank = max(n*m, m*p, p*n)  # Lower bound = 30
    
    for current_rank in range(min_rank, best_rank):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            U = np.random.randn(n * m, current_rank)
            V = np.random.randn(m * p, current_rank)
            W = np.random.randn(p * n, current_rank)
            
            try:
                from scipy.optimize import least_squares
                
                len_U = n * m * current_rank
                len_V = m * p * current_rank
                len_W = p * n * current_rank
                
                def fun(params):
                    U_flat = params[:len_U]
                    V_flat = params[len_U:len_U+len_V]
                    W_flat = params[len_U+len_V:]
                    U = U_flat.reshape((n*m, current_rank))
                    V = V_flat.reshape((m*p, current_rank))
                    W = W_flat.reshape((p*n, current_rank))
                    return compute_residuals(U, V, W, current_rank)
                
                params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result = least_squares(fun, params_init, method='lm', max_nfev=300)
                
                U_res = result.x[:len_U].reshape((n*m, current_rank))
                V_res = result.x[len_U:len_U+len_V].reshape((m*p, current_rank))
                W_res = result.x[len_U+len_V:].reshape((p*n, current_rank))
                
                if check_exact(U_res, V_res, current_rank):
                    best_rank = current_rank
                    best_U = U_res
                    best_V = V_res
                    best_W = W_res
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            except Exception:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
