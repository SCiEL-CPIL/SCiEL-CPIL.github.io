import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
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
        """Compute the residual for the tensor decomposition (vectorized)."""
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
    
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = x[(n*m+m*p)*rank:].reshape(p*n, rank)
        return U, V, W
    
    best_solution = (U17, V17, W17)
    best_res = residual(U17, V17, W17)
    
    # Try to find rank-16 solution using L-BFGS-B
    R = 16
    
    def try_lbfgs(init_U, init_V, init_W, max_iter=500, rank=R):
        nonlocal best_res, best_solution
        x0 = flatten(init_U, init_V, init_W)
        
        def fun(x):
            U, V, W = unflatten(x, rank)
            return residual(U, V, W)
        
        def jac(x):
            U, V, W = unflatten(x, rank)
            Ug, Vg, Wg = residual_grad(U, V, W)
            return flatten(Ug, Vg, Wg)
        
        try:
            result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'gtol': 1e-10})
            if result.fun < best_res:
                U, V, W = unflatten(result.x, rank)
                best_res = result.fun
                best_solution = (U.copy(), V.copy(), W.copy())
            return result.fun
        except Exception:
            return np.inf
    
    # Try each column removal with L-BFGS-B
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U = U17[:, cols]
        V = V17[:, cols]
        W = W17[:, cols]
        
        try_lbfgs(U, V, W, max_iter=400, rank=16)
        
        if best_res < 1e-10:
            break
    
    # Try random initializations with L-BFGS-B - more diverse
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Use different scales for diversity
        scale = 0.2 + 0.1 * (attempt % 5)
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale
        
        try_lbfgs(U, V, W, max_iter=300, rank=16)
        
        if best_res < 1e-10:
            break
    
    # If we found a good rank-16 solution, try to round to half-integers
    if best_res < 1e-6:
        U_best, V_best, W_best = best_solution
        # Try multiple rounding strategies
        best_round_res = np.inf
        best_rounded = None
        
        for factor in [2, 4]:
            U_round = np.round(U_best * factor) / factor
            V_round = np.round(V_best * factor) / factor
            W_round = np.round(W_best * factor) / factor
            
            res_round = residual(U_round, V_round, W_round)
            if res_round < best_round_res:
                best_round_res = res_round
                best_rounded = (U_round, V_round, W_round)
        
        if best_rounded is not None and best_round_res < 1e-6:
            U_round, V_round, W_round = best_rounded
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
