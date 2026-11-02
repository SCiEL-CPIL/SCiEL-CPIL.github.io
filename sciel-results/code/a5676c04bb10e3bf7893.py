import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Pre-compute indices for residuals
    # Residuals are for all i, k, j1, j2
    # Condition: sum_c W[k,i,c] * U[i,j1,c] * V[j2,k,c] == delta[j1, j2]
    
    def compute_residuals(params, r):
        # params is a flat array of length r * (n*m + m*p + p*n)
        # Split into U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U_flat = params[:r * size_u]
        V_flat = params[r * size_u : r * size_u + r * size_v]
        W_flat = params[r * size_u + r * size_v : r * size_u + r * size_v + r * size_w]
        
        U = U_flat.reshape(r, size_u)
        V = V_flat.reshape(r, size_v)
        W = W_flat.reshape(r, size_w)
        
        # Transpose for easier indexing?
        # U[i, j, c] -> U.T[c, i*m+j]
        # V[j, k, c] -> V.T[c, j*p+k]
        # W[k, i, c] -> W.T[c, k*n+i]
        
        residuals = []
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        # Sum over c
                        # U index: i*m + j1
                        # V index: j2*p + k
                        # W index: k*n + i
                        u_val = U[:, i * m + j1]
                        v_val = V[:, j2 * p + k]
                        w_val = W[:, k * n + i]
                        
                        dot = np.dot(np.dot(u_val, v_val), w_val)
                        target = 1.0 if j1 == j2 else 0.0
                        residuals.append(dot - target)
        return np.array(residuals)

    def objective(params, r):
        res = compute_residuals(params, r)
        return np.dot(res, res)

    def snap_and_verify(U, V, W, r):
        # Check if entries are close to integers or half-integers
        # and if the decomposition is exact
        def is_valid_matrix(M):
            for val in M:
                # Check if close to k/2 for some integer k
                # 2*val should be close to integer
                if abs(2 * val - round(2 * val)) > 1e-5:
                    return False
            return True
        
        if not (is_valid_matrix(U) and is_valid_matrix(V) and is_valid_matrix(W)):
            return None
            
        # Snap
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        
        # Verify exactness
        # Recompute residuals with snapped values
        # We need to map back to flat params
        U_flat = U_snap.flatten()
        V_flat = V_snap.flatten()
        W_flat = W_snap.flatten()
        params = np.concatenate([U_flat, V_flat, W_flat])
        
        res = compute_residuals(params, r)
        if np.max(np.abs(res)) < 1e-9:
            # Convert to list of lists
            U_list = U_snap.reshape(r, n * m).T.tolist() # Shape (n*m, r)
            V_list = V_snap.reshape(r, m * p).T.tolist() # Shape (m*p, r)
            W_list = W_snap.reshape(r, p * n).T.tolist() # Shape (p*n, r)
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    # Search for minimal rank
    # Start from schoolbook and go down
    # Given time constraints, we might not search all the way down if it's slow
    # But 105 is small.
    
    best_result = None
    # We know schoolbook works
    # Let's try to find a better one
    
    # Optimization settings
    # We will try ranks from 105 down to 50
    # For each rank, try a few random initializations
    
    # To save time, we can use the schoolbook solution as a base for high ranks
    # But for lower ranks, random is better
    
    # Let's try a few ranks
    # If we find a solution, we return it.
    # We prioritize lower ranks.
    
    # However, finding a solution for a specific rank is hard.
    # Maybe we should just try to optimize the schoolbook solution by projecting to lower rank?
    # No, that doesn't work.
    
    # Let's try ranks: 105, 100, 95, ..., down to 50.
    # Time budget: 110s.
    
    ranks_to_try = list(range(105, 49, -1))
    
    for r in ranks_to_try:
        if time.time() - start_time > 100:
            break
            
        # Try a few restarts for this rank
        restarts = 5
        for _ in range(restarts):
            if time.time() - start_time > 100:
                break
                
            # Initialize
            # Random normal
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = r * (size_u + size_v + size_w)
            
            # Use small random initialization
            x0 = np.random.randn(total_size) * 0.1
            
            # Run optimization
            try:
                res = minimize(objective, x0, args=(r,), method='BFGS', 
                               options={'maxiter': 100, 'disp': False})
                if res.success or res.fun < 1e-6:
                    # Split result
                    U_flat = res.x[:r * size_u]
                    V_flat = res.x[r * size_u : r * size_u + r * size_v]
                    W_flat = res.x[r * size_u + r * size_v : r * size_u + r * size_v + r * size_w]
                    
                    U = U_flat.reshape(r, size_u)
                    V = V_flat.reshape(r, size_v)
                    W = W_flat.reshape(r, size_w)
                    
                    snapped = snap_and_verify(U, V, W, r)
                    if snapped:
                        return snapped
            except Exception:
                pass
    
    # If no better solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
