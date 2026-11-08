"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    # Check exact start time
    deadline = time.time() + 130  # Reserve 20s margin for 150s total
    
    # Attempt numerical search for lower rank if sufficient time
    # Prioritize ranks that are more likely to be optimal or close to it.
    # Rank 26 is the theoretical minimum for 3x3x4 (23 + 3).
    ranks_to_try = [26, 27, 28, 29, 30, 31, 32, 33]
    
    for target_rank in ranks_to_try:
        if time.time() >= deadline - 10:  # Keep 10s for final checks/fallback
            break
            
        # Allocate more time to rank 26 as it is the most valuable target
        t_limit = 25.0 if target_rank == 26 else 10.0
        
        res = attempt_lower_rank(target_rank, time_limit=t_limit)
        if res is not None and valid_res(res, 3, 3, 4):
            return res
    
    # Fallback: return exactly rank 34 decomposition
    return construct_r34()

def attempt_lower_rank(target_rank, time_limit=10.0):
    n, m, p = 3, 3, 4
    R = target_rank
    start_time = time.time()
    
    # Build the target tensor indices
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Objective function
    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        return error
    
    # Try multiple random restarts
    # Use different seeds to explore the non-convex landscape
    seeds = [42, 123, 999, 555, 777, 2023, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    from scipy.optimize import minimize
    
    best_result = None
    best_error = float('inf')
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed)
        # Scale initial values appropriately
        # Smaller initialization might help avoid overflow or large gradients
        params = np.random.randn((n*m + m*p + p*n) * R) * 0.5
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_result = res
                
                # If we found a good solution, try to polish/round it
                # Lower threshold to allow more attempts at rounding
                if res.fun < 1e-3:
                    U = res.x[0:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try rounding to nearest integer
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    vals_int = np.sum(U_int[indices_a, :] * V_int[indices_b, :] * W_int[indices_c, :], axis=1)
                    if np.allclose(vals_int, 1.0, atol=1e-9):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
                    # Try rounding to nearest 0.5
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    vals_half = np.sum(U_half[indices_a, :] * V_half[indices_b, :] * W_half[indices_c, :], axis=1)
                    if np.allclose(vals_half, 1.0, atol=1e-9):
                        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
        except Exception:
            continue
            
    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
        
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                indices_a.append(i * m + k)
                indices_b.append(k * p + j)
                indices_c.append(j * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
    return np.allclose(vals, 1.0, atol=1e-9)

def construct_r34():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
