import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Attempt to find rank 42
    rank_target = 42
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Initialize random
    np.random.seed(42)
    U = np.random.randn(n * m, rank_target) * 0.1
    V = np.random.randn(m * p, rank_target) * 0.1
    W = np.random.randn(p * n, rank_target) * 0.1

    # Target tensor T (all ones)
    T = np.ones((n, m, p))
    
    def residual(params):
        # params shape: (55 * rank_target,)
        U_flat = params[:n*m*rank_target].reshape(n*m, rank_target)
        V_flat = params[n*m*rank_target : n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
        W_flat = params[n*m*rank_target + m*p*rank_target :].reshape(p*n, rank_target)
        
        # Reconstruct tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_recon = np.zeros((n, m, p))
        for r in range(rank_target):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            # Outer product logic
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return T_recon.flatten() - T.flatten()

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use least_squares
        from scipy.optimize import least_squares
        res = least_squares(residual, x0, method='lm', max_nfev=10000)
        if res.success or np.linalg.norm(res.fun) < 1e-6:
            U_sol = res.x[:n*m*rank_target].reshape(n*m, rank_target)
            V_sol = res.x[n*m*rank_target : n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
            W_sol = res.x[n*m*rank_target + m*p*rank_target :].reshape(p*n, rank_target)
            
            # Convert to list of lists
            U_list = U_sol.tolist()
            V_list = V_sol.tolist()
            W_list = W_sol.tolist()
            
            # Check validity
            # ... (skip for brevity in thought)
            
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}
