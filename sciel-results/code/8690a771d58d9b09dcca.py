"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-6:
                            return False
        return True
    
    # Loss function for optimization: sum of squared errors
    def loss(params, target_rank):
        U = params[:n * m * target_rank].reshape(n * m, target_rank)
        V = params[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W = params[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        total_loss = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(target_rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        total_loss += (actual - expected) ** 2
        return total_loss
    
    # Try to find lower rank decompositions using optimization
    for target_rank in range(rank_school - 1, max(30, rank_school - 30), -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random perturbations of schoolbook
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead for optimization (no gradient needed)
            result = minimize(loss, params, args=(target_rank,), method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt = result.x[:n * m * target_rank].reshape(n * m, target_rank)
                V_opt = result.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                W_opt = result.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                
                if verify(U_opt, V_opt, W_opt, target_rank):
                    best_rank = target_rank
                    best_U = U_opt.copy()
                    best_V = V_opt.copy()
                    best_W = W_opt.copy()
                    break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
