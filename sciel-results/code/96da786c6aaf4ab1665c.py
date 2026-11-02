import numpy as np
import time
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook initialization (rank 45)
    rank_seed = n * m * p
    U = np.zeros((n * m, rank_seed))
    V = np.zeros((m * p, rank_seed))
    W = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    current_rank = rank_seed
    
    def check_valid(U, V, W, rank, tol=1e-5):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Precompute indices for speed
    idx_i = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    idx_j = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    idx_k = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    # Vectorized residual computation
    def compute_residual_vec(params, rank):
        U_r = params[:n*m*rank].reshape((n*m, rank))
        V_r = params[n*m*rank:n*m*rank + m*p*rank].reshape((m*p, rank))
        W_r = params[n*m*rank + m*p*rank:].reshape((p*n, rank))
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 for all i,j,k
        
        # Build the full tensor via einsum or loops
        # For speed with small dimensions, use loops but vectorize over r
        residuals = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    residuals[idx] = val - 1.0
        return residuals
    
    # Try greedy removal with local optimization
    max_iterations = 50
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        if not check_valid(U, V, W, current_rank, tol=1e-4):
            break
            
        removed = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing term r
            mask = [x for x in range(current_rank) if x != r]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            new_rank = current_rank - 1
            
            # Check if already valid
            if check_valid(U_new, V_new, W_new, new_rank, tol=1e-4):
                U, V, W = U_new, V_new, W_new
                current_rank = new_rank
                removed = True
                break
            
            # Try local optimization to recover validity
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            try:
                result = least_squares(
                    compute_residual_vec, x0, args=(new_rank,),
                    method='lm',
                    max_nfev=500,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                if result.cost < 1e-10:
                    U_opt = result.x[:n*m*new_rank].reshape((n*m, new_rank))
                    V_opt = result.x[n*m*new_rank:n*m*new_rank + m*p*new_rank].reshape((m*p, new_rank))
                    W_opt = result.x[n*m*new_rank + m*p*new_rank:].reshape((p*n, new_rank))
                    if check_valid(U_opt, V_opt, W_opt, new_rank, tol=1e-4):
                        U, V, W = U_opt, V_opt, W_opt
                        current_rank = new_rank
                        removed = True
                        break
            except Exception:
                pass
        
        if not removed:
            # Try a different strategy: optimize the current rank to see if we can find a sparser solution
            # Or try removing multiple terms at once
            # For now, break out
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
