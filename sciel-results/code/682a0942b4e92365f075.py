"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 32
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def compute_error(U, V, W):
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0)**2
        return np.sqrt(err)
    
    def als(U, V, W, r, max_iters=300, tol=1e-12):
        U = U[:, :r].copy()
        V = V[:, :r].copy()
        W = W[:, :r].copy()
        
        U += np.random.randn(*U.shape) * 1e-8
        V += np.random.randn(*V.shape) * 1e-8
        W += np.random.randn(*W.shape) * 1e-8
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            new_U = np.zeros((n*m, r))
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=1e-15)
                    except:
                        x = np.linalg.pinv(M) @ np.ones(p)
                    new_U[i*m+j, :] = x
            
            new_V = np.zeros((m*p, r))
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=1e-15)
                    except:
                        x = np.linalg.pinv(M) @ np.ones(n)
                    new_V[j*p+k, :] = x
            
            new_W = np.zeros((p*n, r))
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=1e-15)
                    except:
                        x = np.linalg.pinv(M) @ np.ones(m)
                    new_W[k*n+i, :] = x
            
            U, V, W = new_U, new_V, new_W
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
        
        return best_U, best_V, best_W, best_err
    
    best_result = None
    
    for target_rank in range(28, 32):
        if time.time() > deadline - 15:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            if target_rank >= 31:
                U_init = U[:, :target_rank]
                V_init = V[:, :target_rank]
                W_init = W[:, :target_rank]
            else:
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            
            U_res, V_res, W_res, err = als(U_init, V_init, W_init, target_rank, max_iters=200)
            
            if err < 1e-8:
                best_result = (U_res, V_res, W_res, target_rank)
                break
        
        if best_result is not None:
            continue
        else:
            break
    
    if best_result is not None:
        U, V, W = best_result[0], best_result[1], best_result[2]
    else:
        # Fallback to schoolbook
        pass
    
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    r.append(rounded)
                else:
                    r.append(float(val))
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
