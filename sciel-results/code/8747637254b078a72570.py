import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _full_error(U, V, W):
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    max_err = 0.0
    I_m = np.eye(m)
    
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(U_np.shape[1]):
                u_vec = U_np[i*m : i*m+m, r]
                v_vec = V_np[:, r] # No, need specific k
                # v_vec[j] = V[j*p + k, r]
                v_vec = V_np[np.arange(m) * p + k, r]
                w_scalar = W_np[k * n + i, r]
                M += w_scalar * np.outer(u_vec, v_vec)
            err = np.max(np.abs(M - I_m))
            if err > max_err:
                max_err = err
    return max_err

def _try_rank_decomposition(target_rank, num_attempts=5, max_iter=500, deadline=None):
    n, m, p = DIMENSIONS
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    I_m = np.eye(m)
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(attempt * 1000 + target_rank)
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        lr = 0.02
        patience = 0
        cur_err = _full_error(U, V, W)
        
        for it in range(max_iter):
            if deadline and time.time() > deadline - 15:
                break
                
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, m))
                    for r in range(target_rank):
                        u_vec = U[i*m : i*m+m, r]
                        v_vec = V[np.arange(m) * p + k, r]
                        w_scalar = W[k * n + i, r]
                        M += w_scalar * np.outer(u_vec, v_vec)
                    
                    res = M - I_m
                    
                    for r in range(target_rank):
                        u_vec = U[i*m : i*m+m, r]
                        v_vec = V[np.arange(m) * p + k, r]
                        w_scalar = W[k * n + i, r]
                        
                        # dL/dU
                        dU[i*m : i*m+m, r] += 2 * w_scalar * (res @ v_vec)
                        
                        # dL/dV
                        dV[np.arange(m) * p + k, r] += 2 * w_scalar * (res.T @ u_vec)
                        
                        # dL/dW
                        dW[k * n + i, r] += 2 * np.sum(res * np.outer(u_vec, v_vec))
            
            U_new = U - lr * dU
            V_new = V - lr * dV
            W_new = W - lr * dW
            
            new_err = _full_error(U_new, V_new, W_new)
            
            if new_err < cur_err:
                U, V, W = U_new, V_new, W_new
                cur_err = new_err
                patience = 0
                if cur_err < best_err:
                    best_err = cur_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                patience += 1
                lr *= 0.8
                if patience > 15:
                    break
            
            if best_err < 1e-9:
                break
                
    if best_err < 1e-8:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    
    for target_rank in range(60, 120):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_decomposition(target_rank, num_attempts=3, max_iter=300, deadline=deadline)
        if result is not None:
            best_result = result
            break
            
    return best_result
# EVOLVE-BLOCK-END
