import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def try_find_better_rank(n, m, p, target_rank, time_limit=100):
    start_time = time.time()
    # Build the target tensor T as a list of (i, j, k) indices where value is 1
    # T_{i, j, k} = 1
    # We need sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r} = 1
    
    # Flatten indices
    # U: (nm, R), V: (mp, R), W: (pn, R)
    
    def residual(params):
        # params: flattened U, V, W
        R = target_rank
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = params[:size_u].reshape(n * m, R)
        V = params[size_u:size_u + size_v].reshape(m * p, R)
        W = params[size_u + size_v:].reshape(p * n, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over r
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    res.append(val - 1.0)
        return res

    # Try a few random initializations
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
            
        R = target_rank
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        total_size = size_u + size_v + size_w
        
        # Random init
        np.random.seed(attempt)
        params0 = np.random.randn(total_size) * 0.1
        
        try:
            result = least_squares(residual, params0, method='lm', max_nfev=1000)
            if result.success and np.max(np.abs(result.fun)) < 1e-5:
                # Check if rounding to half-integers works
                U = result.x[:size_u].reshape(n * m, R)
                V = result.x[size_u:size_u + size_v].reshape(m * p, R)
                W = result.x[size_u + size_v:].reshape(p * n, R)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_r[u_idx, :], np.multiply(V_r[v_idx, :], W_r[w_idx, :]))
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better rank solution
    # Known optimum for 3x3x7 is 27
    better = try_find_better_rank(n, m, p, 27, time_limit=100)
    if better:
        return {"U": better[0], "V": better[1], "W": better[2]}
        
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
