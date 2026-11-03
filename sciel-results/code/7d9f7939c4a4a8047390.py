import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s
    
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
    
    # Precompute index tuples for efficient residual/gradient
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    indices = np.array(indices)
    u_idx = indices[:, 0]
    v_idx = indices[:, 1]
    w_idx = indices[:, 2]
    n_entries = len(indices)
    
    def residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        vals = np.einsum('tr,sr,tr->t', U[u_idx], V[v_idx], W[w_idx])
        return np.sum((vals - 1.0) ** 2)
    
    def residual_and_grad(U, V, W):
        """Compute residual and gradients analytically (vectorized)."""
        vals = np.einsum('tr,sr,tr->t', U[u_idx], V[v_idx], W[w_idx])
        err = vals - 1.0
        res = np.sum(err ** 2)
        
        # Precompute products
        vw = V[v_idx] * W[w_idx]  # (n_entries, R)
        uw = U[u_idx] * W[w_idx]  # (n_entries, R)
        uv = U[u_idx] * V[v_idx]  # (n_entries, R)
        
        # Scatter-add gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        np.add.at(gU, (u_idx, slice(None)), 2.0 * err[:, None] * vw)
        np.add.at(gV, (v_idx, slice(None)), 2.0 * err[:, None] * uw)
        np.add.at(gW, (w_idx, slice(None)), 2.0 * err[:, None] * uv)
        
        return res, np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    def try_optimize(U_start, V_start, W_start, maxiter=200, maxfun=3000, gtol=1e-8):
        """Try to optimize the given initial point."""
        U_size = U_start.size
        V_size = V_start.size
        
        def objective(x):
            U = x[:U_size].reshape(U_start.shape)
            V = x[U_size:U_size+V_size].reshape(V_start.shape)
            W = x[U_size+V_size:].reshape(W_start.shape)
            return residual(U, V, W)
        
        def objective_grad(x):
            U = x[:U_size].reshape(U_start.shape)
            V = x[U_size:U_size+V_size].reshape(V_start.shape)
            W = x[U_size+V_size:].reshape(W_start.shape)
            return residual_and_grad(U, V, W)
        
        x0 = np.concatenate([U_start.flatten(), V_start.flatten(), W_start.flatten()])
        
        try:
            result = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'maxfun': maxfun, 'gtol': gtol})
            return result.fun, result.x
        except Exception:
            return float('inf'), None
    
    def verify_and_round(U, V, W):
        """Verify the solution and try to round to half-integers."""
        res = residual(U, V, W)
        if res < 1e-6:
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            res_round = residual(U_round, V_round, W_round)
            if res_round < 1e-9:
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        return None
    
    # Try to find rank-16 solution
    R = 16
    best_res = residual(U17, V17, W17)
    best_solution = (U17.copy(), V17.copy(), W17.copy())
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Try each column removal with multiple restarts
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U0 = U17[:, cols].copy()
        V0 = V17[:, cols].copy()
        W0 = W17[:, cols].copy()
        
        # Multiple restarts: original + perturbed
        n_restarts = 2
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                U_start = U0.copy()
                V_start = V0.copy()
                W_start = W0.copy()
            else:
                # Add small random perturbation to escape local minima
                scale = 0.05
                U_start = U0 + scale * rng.standard_normal(U0.shape)
                V_start = V0 + scale * rng.standard_normal(V0.shape)
                W_start = W0 + scale * rng.standard_normal(W0.shape)
            
            res, x = try_optimize(U_start, V_start, W_start, maxiter=200, maxfun=3000, gtol=1e-8)
            
            if res < best_res:
                best_res = res
                U_best = x[:U0.size].reshape(U0.shape)
                V_best = x[U0.size:U0.size+V0.size].reshape(V0.shape)
                W_best = x[U0.size+V0.size:].reshape(W0.shape)
                best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
                
                result = verify_and_round(U_best, V_best, W_best)
                if result is not None:
                    return result
    
    # Strategy 2: Random initializations for rank-16
    n_random_tries = 10
    for trial in range(n_random_tries):
        if time.time() > deadline:
            break
        
        U_rand = rng.standard_normal((n * m, R)) * 0.5
        V_rand = rng.standard_normal((m * p, R)) * 0.5
        W_rand = rng.standard_normal((p * n, R)) * 0.5
        
        res, x = try_optimize(U_rand, V_rand, W_rand, maxiter=150, maxfun=2000, gtol=1e-7)
        
        if res < best_res:
            best_res = res
            U_best = x[:U_rand.size].reshape(U_rand.shape)
            V_best = x[U_rand.size:U_rand.size+V_rand.size].reshape(V_rand.shape)
            W_best = x[U_rand.size+V_rand.size:].reshape(W_rand.shape)
            best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
            
            result = verify_and_round(U_best, V_best, W_best)
            if result is not None:
                return result
    
    # Final high-precision polish pass on the best rank-16 solution found
    if best_res < 1e-4 and time.time() < deadline - 20:
        U_best, V_best, W_best = best_solution
        res, x = try_optimize(U_best, V_best, W_best, maxiter=500, maxfun=10000, gtol=1e-12)
        
        if res < best_res:
            best_res = res
            U_best = x[:U_best.size].reshape(U_best.shape)
            V_best = x[U_best.size:U_best.size+V_best.size].reshape(V_best.shape)
            W_best = x[U_best.size+V_best.size:].reshape(W_best.shape)
            best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
            
            result = verify_and_round(U_best, V_best, W_best)
            if result is not None:
                return result

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
