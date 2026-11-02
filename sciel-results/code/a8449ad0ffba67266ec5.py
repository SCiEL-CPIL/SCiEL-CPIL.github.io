import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
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
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Verify schoolbook is correct
    U3 = np.array(U_sb).reshape(n, m, rank_sb)
    V3 = np.array(V_sb).reshape(m, p, rank_sb)
    W3 = np.array(W_sb).reshape(p, n, rank_sb)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if np.max(np.abs(T - 1.0)) < 1e-9:
        # Valid schoolbook
        pass
    
    # Try to find lower rank using random initialization + optimization
    # The bilinear complexity of 3x3x7 multiplication:
    # Lower bound: max(9, 9, 21) = 21
    # Upper bound: 63 (schoolbook)
    # Known results suggest rank around 21-30 might be achievable
    
    for target_rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        R = target_rank
        np.random.seed(42 + R)
        
        # Multiple random restarts
        n_restarts = 3 if R <= 30 else 2
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(100 + R * 10 + restart)
            
            # Initialize with small random values
            U0 = np.random.randn(n * m, R) * 0.5
            V0 = np.random.randn(m * p, R) * 0.5
            W0 = np.random.randn(p * n, R) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n, m, R)
                V = x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
                W = x[n*m*R+m*p*R:].reshape(p, n, R)
                T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                residual = T - 1.0
                return np.sum(residual**2)
            
            # Use L-BFGS-B with time-aware iteration limit
            t_opt_start = time.time()
            max_iter = 500
            try:
                from scipy.optimize import minimize
                result = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            except:
                break
                
            if time.time() > deadline:
                break
                
            if result.fun < 1e-6:
                # Found a valid decomposition
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Try to round to half-integers or integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                U3_r = U_rounded.reshape(n, m, R)
                V3_r = V_rounded.reshape(m, p, R)
                W3_r = W_rounded.reshape(p, n, R)
                T_r = np.einsum('ijr,jkr,kir->ijk', U3_r, V3_r, W3_r)
                
                if np.max(np.abs(T_r - 1.0)) < 1e-6:
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = R
                    found = True
                    break
                else:
                    # Return unrounded if rounding fails but tolerance is ok
                    # The task says integer/half-integer entries required
                    # So we must round. If it doesn't work, skip this rank.
                    pass
        
    return best_result
