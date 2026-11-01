import numpy as np
import time
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS

    def verify(U_mat, V_mat, W_mat, r):
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            C_decomp = np.zeros((n, p))
            for t in range(r):
                u_t = U_mat[:, t].reshape(n, m)
                v_t = V_mat[:, t].reshape(m, p)
                w_t = W_mat[:, t].reshape(p, n)
                C_decomp += np.sum(A_test * u_t) * np.sum(B_test * v_t) * w_t.T
            if not np.allclose(C_test, C_decomp, atol=1e-9):
                return False
        return True

    def find_decomposition(r, max_iters=200):
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residual(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:2*n*m*r].reshape(m*p, r)
            W = x[2*n*m*r:].reshape(p*n, r)
            
            T_comp = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_comp[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            return T_comp.flatten() - np.ones(n*m*p)
            
        def objective(x):
            r_vec = residual(x)
            return np.sum(r_vec ** 2)
            
        try:
            # Use least_squares for better conditioning
            res = least_squares(residual, x0, method='lm', max_nfev=max_iters)
            if res.cost < 1e-12:
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r:2*n*m*r].reshape(m*p, r)
                W_opt = x_opt[2*n*m*r:].reshape(p*n, r)
                
                if verify(U_opt, V_opt, W_opt, r):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
            
        # Fallback to minimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'gtol': 1e-12})
            if result.fun < 1e-12:
                x_opt = result.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r:2*n*m*r].reshape(m*p, r)
                W_opt = x_opt[2*n*m*r:].reshape(p*n, r)
                if verify(U_opt, V_opt, W_opt, r):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
            
        return None

    # Try ranks in increasing order
    for r in [25, 26, 27, 28]:
        if time.time() > DEADLINE - 20:
            break
        res = find_decomposition(r)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook rank 30
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col_sb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col_sb] = 1.0
                V_sb[j * p + k, col_sb] = 1.0
                W_sb[k * n + i, col_sb] = 1.0
                col_sb += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
