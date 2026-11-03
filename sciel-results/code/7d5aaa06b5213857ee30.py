import time
import random
import math
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    def get_tensor(n, m, p):
        """Return the multiplication tensor as a dictionary of non-zero entries."""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[(a, b, c)] = 1
        return T

    def validate(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for c_idx in range(rank):
                        s += U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                    if abs(s - 1) > tol:
                        return False
        
        # Check zeros
        valid_indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_indices.add((i * m + j, j * p + k, k * n + i))
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in valid_indices:
                        continue
                    s = 0
                    for c_idx in range(rank):
                        s += U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                    if abs(s) > tol:
                        return False
        return True

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_find_decomposition(n, m, p, target_rank, time_limit=30):
        """Try to find a decomposition of rank target_rank."""
        start_time = time.time()
        T = get_tensor(n, m, p)
        
        def objective(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            err = 0
            for (a, b, c), val in T.items():
                s = np.dot(U[a], np.multiply(V[b], W[c]))
                err += (s - val) ** 2
            # Add penalty for large values
            err += 0.01 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return err

        best_sol = None
        best_err = float('inf')
        
        # Use more restarts if time permits
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_sol = res.x
                
                if best_err < 1e-6:
                    U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Try rounding to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also try rounding the current best if it's good enough
            if best_err < 1e-4:
                U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    # Strategy: Try to find a rank r decomposition for 3x3x6 directly
    # The schoolbook rank is 54. We try to find lower ranks.
    # Known minimum rank for 3x3x3 is 23, so 3x3x6 might be around 46.
    
    for r in [46, 47, 48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(n, m, p, r, time_limit=15)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
