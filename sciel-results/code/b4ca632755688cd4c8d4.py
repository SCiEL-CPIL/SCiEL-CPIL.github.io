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
    
    def residual_grad(U, V, W):
        """Compute the gradient of the residual."""
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = val - 1.0
                    VW = V[v_idx, :] * W[w_idx, :]
                    U_grad[u_idx, :] += 2 * diff * VW
                    V_grad[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
                    W_grad[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
        return U_grad, V_grad, W_grad
    
    def pack(U, V, W):
        """Pack U, V, W into a single vector."""
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unpack(x):
        """Unpack a vector into U, V, W."""
        U_size = n * m * 16
        V_size = m * p * 16
        W_size = p * n * 16
        U = x[:U_size].reshape(n * m, 16)
        V = x[U_size:U_size + V_size].reshape(m * p, 16)
        W = x[U_size + V_size:].reshape(p * n, 16)
        return U, V, W
    
    def objective(x):
        """Objective function for optimizer."""
        U, V, W = unpack(x)
        return residual(U, V, W)
    
    def objective_grad(x):
        """Gradient for optimizer."""
        U, V, W = unpack(x)
        Ug, Vg, Wg = residual_grad(U, V, W)
        return np.concatenate([Ug.flatten(), Vg.flatten(), Wg.flatten()])
    
    best_solution = (U17, V17, W17)
    best_res = residual(U17, V17, W17)
    best_rank = 17
    
    # Try to find rank-16 solution using L-BFGS-B
    R = 16
    
    # Try each column removal with L-BFGS-B
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U_init = U17[:, cols]
        V_init = V17[:, cols]
        W_init = W17[:, cols]
        
        x0 = pack(U_init, V_init, W_init)
        
        try:
            result = minimize(
                objective, x0, jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_res:
                U_opt, V_opt, W_opt = unpack(result.x)
                best_res = result.fun
                best_solution = (U_opt, V_opt, W_opt)
                best_rank = 16
                
                if best_res < 1e-10:
                    break
        except Exception:
            continue
    
    # Try random initializations with L-BFGS-B
    np.random.seed(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        U_init = np.random.randn(n * m, R) * 0.5
        V_init = np.random.randn(m * p, R) * 0.5
        W_init = np.random.randn(p * n, R) * 0.5
        
        x0 = pack(U_init, V_init, W_init)
        
        try:
            result = minimize(
                objective, x0, jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_res:
                U_opt, V_opt, W_opt = unpack(result.x)
                best_res = result.fun
                best_solution = (U_opt, V_opt, W_opt)
                best_rank = 16
                
                if best_res < 1e-10:
                    break
        except Exception:
            continue
    
    # If we found a good rank-16 solution, try to round to half-integers
    if best_rank == 16 and best_res < 1e-6:
        U_best, V_best, W_best = best_solution
        # Round to nearest half-integer
        U_round = np.round(U_best * 2) / 2
        V_round = np.round(V_best * 2) / 2
        W_round = np.round(W_best * 2) / 2
        
        res_round = residual(U_round, V_round, W_round)
        if res_round < 1e-6:
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        # Try rounding to integers
        U_int = np.round(U_best)
        V_int = np.round(V_best)
        W_int = np.round(W_best)
        
        res_int = residual(U_int, V_int, W_int)
        if res_int < 1e-6:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
