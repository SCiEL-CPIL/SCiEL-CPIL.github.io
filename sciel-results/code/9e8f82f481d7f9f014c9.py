import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
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
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    def error_function(params, r):
        """Calculate squared error for given parameters and rank."""
        num_u = n * m * r
        num_v = m * p * r
        num_w = p * n * r
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u + num_v]
        W_flat = params[num_u + num_v:num_u + num_v + num_w]
        
        U = U_flat.reshape((n * m, r))
        V = V_flat.reshape((m * p, r))
        W = W_flat.reshape((p * n, r))
        
        err = 0.0
        # Vectorized calculation for speed
        # For each c, we have U[:,c], V[:,c], W[:,c]
        # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is a bit hard to fully vectorize across i,j,k without loops due to index mapping.
        # But n*m*p is only 30.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Dot product of the rows
                    val = np.dot(np.dot(U[u_idx], V[v_idx]), W[w_idx])
                    diff = val - 1.0
                    err += diff * diff
        return err

    def try_rank(r, num_restarts=50):
        """Try to find a valid decomposition of rank r."""
        if time.time() > deadline:
            return None
            
        np.random.seed(42) # Fixed seed for reproducibility within rank? 
        # Actually, we want different seeds for different restarts.
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale to avoid overflow/underflow
            scale = 1.0
            U_init = np.random.randn(n * m, r) * scale
            V_init = np.random.randn(m * p, r) * scale
            W_init = np.random.randn(p * n, r) * scale
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimization
            try:
                res = minimize(
                    lambda x: error_function(x, r),
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if res.fun < 1e-8:
                    # Extract optimized matrices
                    num_u = n * m * r
                    num_v = m * p * r
                    num_w = p * n * r
                    
                    U_opt = res.x[:num_u].reshape((n * m, r))
                    V_opt = res.x[num_u:num_u + num_v].reshape((m * p, r))
                    W_opt = res.x[num_u + num_v:num_u + num_v + num_w].reshape((p * n, r))
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2.0
                    V_snap = np.round(V_opt * 2) / 2.0
                    W_snap = np.round(W_opt * 2) / 2.0
                    
                    # Verify exactness
                    if verify(U_snap, V_snap, W_snap, r):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
            except Exception:
                continue
                
        return None

    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(np.dot(U[u_idx], V[v_idx]), W[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks from lower bound upwards
    # Lower bound for 2x3x5 is 15.
    # We try 15, 16, 17, 18, 19, 20.
    # If we fail, we return schoolbook.
    
    for r in range(15, 26):
        if time.time() > deadline:
            break
            
        result = try_rank(r, num_restarts=20)
        if result:
            best_result = result
            best_rank = r
            break # Found minimal? Not necessarily, but we stop to save time if we found a low one?
            # Actually, we should keep going if we want the absolute minimum, but 15 is the lower bound.
            # If we find 15, it's optimal. If we find 16, maybe 15 exists?
            # Given time constraints, finding 15 is hard. Finding 20 is easier.
            # Let's try to find the lowest possible within time.
            # If we find 20, we might still have time to try 19?
            # But the prompt says "rank-minimal".
            # Let's try ranks in order and break on first success.
            # This assumes we find the minimal one first.
            # If 15 is impossible, we find 16, etc.

    return best_result
