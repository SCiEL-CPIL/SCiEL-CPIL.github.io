import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_error = 0.0

    # Precompute index arrays for vectorized error computation
    U_indices = np.array([[i * m + j for j in range(m)] for i in range(n)])  # (n, m)
    V_indices = np.array([[j * p + k for j in range(m)] for k in range(p)])  # (p, m)
    
    def compute_error(U, V, W):
        """Vectorized error computation. U:(n*m,R), V:(m*p,R), W:(p*n,R)"""
        err = 0.0
        for i in range(n):
            for k in range(p):
                u_slice = U[U_indices[i], :]  # (m, R)
                v_slice = V[V_indices[k], :]  # (m, R)
                w_vals = W[k * n + i, :]  # (R,)
                acc = (u_slice * w_vals[None, :]) @ v_slice.T  # (m, m)
                err += np.sum((acc - np.eye(m)) ** 2)
        return err

    # Try to find a lower rank decomposition
    for R in range(15, 40):
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            from scipy.optimize import minimize
            
            def objective(params):
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_error(U_flat, V_flat, W_flat)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                if result.fun < 1e-6:
                    U_opt = result.x[:n*m*R].reshape(n*m, R)
                    V_opt = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = result.x[n*m*R + m*p*R :].reshape(p*n, R)
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded)
                    if err_rounded < 1e-4:
                        best_rank = R
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_error = err_rounded
                        if err_rounded < 1e-6:
                            return {"U": best_U, "V": best_V, "W": best_W}
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
