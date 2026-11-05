import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def run_als_vectorized(rank, max_iters=200, tol=1e-9, seed=42):
        np.random.seed(seed)
        # Initialize U, V, W
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    V_slice = V[j*p : (j+1)*p, :]
                    W_slice = W[i::n, :]
                    M_u = V_slice * W_slice
                    target = T[i, j, :]
                    u_new, residuals, rank_u, s = np.linalg.lstsq(M_u, target, rcond=None)
                    U[i*m+j, :] = u_new
            
            # Update V
            for j in range(m):
                for k in range(p):
                    U_slice = U[j::m, :]
                    W_slice = W[k*n : (k+1)*n, :]
                    M_v = U_slice * W_slice
                    target = T[:, j, k]
                    v_new, residuals, rank_v, s = np.linalg.lstsq(M_v, target, rcond=None)
                    V[j*p+k, :] = v_new
            
            # Update W
            for k in range(p):
                for i in range(n):
                    U_slice = U[i*m : (i+1)*m, :]
                    V_slice = V[k::p, :]
                    M_w = U_slice * V_slice
                    target = T[i, :, k]
                    w_new, residuals, rank_w, s = np.linalg.lstsq(M_w, target, rcond=None)
                    W[k*n+i, :] = w_new
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (T[i, j, k] - val) ** 2
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err

    def try_rank(rank, seeds=5):
        best_err = float('inf')
        best_factors = None
        for seed in range(seeds):
            U, V, W, err = run_als_vectorized(rank, max_iters=50, seed=seed)
            if err < best_err:
                best_err = err
                best_factors = (U, V, W)
            if err < 1e-6:
                break
        return best_err, best_factors

    start_time = time.time()
    time_limit = 110
    
    for rank in range(71, 59, -1):
        if time.time() - start_time > time_limit:
            break
        
        err, factors = try_rank(rank, seeds=5)
        if factors is None:
            continue
            
        U, V, W = factors
        
        if err < 1e-6:
            # Check for integer/half-integer
            U_int = np.rint(U)
            V_int = np.rint(V)
            W_int = np.rint(W)
            
            int_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                        int_err += (T[i, j, k] - val) ** 2
                        
            if int_err < 1e-6:
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
            U_half = np.rint(U * 2) / 2
            V_half = np.rint(V * 2) / 2
            W_half = np.rint(W * 2) / 2
            
            half_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_half[i*m+j, :], V_half[j*p+k, :] * W_half[k*n+i, :])
                        half_err += (T[i, j, k] - val) ** 2
                        
            if half_err < 1e-6:
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }

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
# EVOLVE-BLOCK-END
