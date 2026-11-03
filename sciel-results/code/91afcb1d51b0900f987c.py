import time
import random
import math
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_global = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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
        # Precompute set of valid indices for faster lookup
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

    # Strategy:
    # 1. Try to find rank 23 for 3x3x3
    # 2. If found, construct rank 46 for 3x3x6
    # 3. If not, try to find rank < 54 for 3x3x6 directly
    
    # First, try to find rank 23 for 3x3x3
    res_333 = try_find_decomposition(3, 3, 3, 23, time_limit=30)
    
    if res_333 is not None:
        # Construct rank 46 for 3x3x6
        U_333 = res_333['U']
        V_333 = res_333['V']
        W_333 = res_333['W']
        
        rank_333 = len(U_333[0])
        
        # For 3x3x6, we can split the p=6 dimension into two p=3 blocks
        # U is (n*m, rank) = (9, 46)
        # V is (m*p, rank) = (18, 46)
        # W is (p*n, rank) = (18, 46)
        
        U_336 = [[0] * (2 * rank_333) for _ in range(9)]
        V_336 = [[0] * (2 * rank_333) for _ in range(18)]
        W_336 = [[0] * (2 * rank_333) for _ in range(18)]
        
        # Block 1: k=0,1,2 (rows 0-2 of B, rows 0-2 of C)
        # Block 2: k=3,4,5 (rows 3-5 of B, rows 3-5 of C)
        
        for c in range(rank_333):
            # Block 1
            for i in range(9):
                U_336[i][c] = U_333[i][c]
            for j in range(3):
                for k in range(3):
                    V_336[j*3 + k][c] = V_333[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_336[k*3 + i][c] = W_333[k*3 + i][c]
            
            # Block 2
            for i in range(9):
                U_336[i][c + rank_333] = U_333[i][c]
            for j in range(3):
                for k in range(3):
                    V_336[(j*3 + 3) + k][c + rank_333] = V_333[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_336[(k*3 + 3) + i][c + rank_333] = W_333[k*3 + i][c]
        
        if validate(U_336, V_336, W_336, 3, 3, 6):
            return {"U": U_336, "V": V_336, "W": W_336}
        
        # If validation fails, fall through to direct search

    # Search for 3x3x6 directly
    # Try ranks 46, 47, ..., 53
    for r in range(46, 54):
        if time.time() > deadline - 10:
            break
        res = try_find_decomposition(3, 3, 6, r, time_limit=15)
        if res is not None:
            return res
            
    # Fallback
    return schoolbook_decomposition(3, 3, 6)
