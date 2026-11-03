import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    # The rank-17 solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    def residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res += (val - 1.0) ** 2
        return res
    
    # Try to find rank-16 solution
    R = 16
    best_res = residual(U17, V17, W17)
    best_solution = (U17.copy(), V17.copy(), W17.copy())
    
    # Try each column removal
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U0 = U17[:, cols].copy()
        V0 = V17[:, cols].copy()
        W0 = W17[:, cols].copy()
        
        # Flatten for optimization
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            U = x[:U0.size].reshape(U0.shape)
            V = x[U0.size:U0.size+V0.size].reshape(V0.shape)
            W = x[U0.size+V0.size:].reshape(W0.shape)
            return residual(U, V, W)
        
        # Use L-BFGS-B for speed and robustness
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'maxfun': 2000})
            
            if result.fun < best_res:
                best_res = result.fun
                U_best = result.x[:U0.size].reshape(U0.shape)
                V_best = result.x[U0.size:U0.size+V0.size].reshape(V0.shape)
                W_best = result.x[U0.size+V0.size:].reshape(W0.shape)
                best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
                
                # If close to zero, try to round to half-integers
                if best_res < 1e-6:
                    U_round = np.round(U_best * 2) / 2
                    V_round = np.round(V_best * 2) / 2
                    W_round = np.round(W_best * 2) / 2
                    
                    res_round = residual(U_round, V_round, W_round)
                    if res_round < 1e-9:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
        except Exception:
            pass

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
