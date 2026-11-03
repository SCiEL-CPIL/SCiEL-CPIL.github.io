import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    n, m, p = 2, 5, 5
    
    # Precompute index mappings for vectorization
    # I, J, K arrays of shape (n, m, p)
    I, J, K = np.indices((n, m, p))
    
    # Map to flat indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I
    
    def compute_residuals(U, V, W):
        """
        Vectorized computation of residuals.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        Returns: (n, m, p) array of residuals.
        """
        R = U.shape[1]
        # Gather rows into 3D arrays of shape (n, m, p, R)
        U_gathered = U[U_idx, :]  # (n, m, p, R)
        V_gathered = V[V_idx, :]  # (n, m, p, R)
        W_gathered = W[W_idx, :]  # (n, m, p, R)
        
        # Compute product sum over R
        # (n, m, p, R) * (n, m, p, R) * (n, m, p, R) -> sum over last axis
        values = np.sum(U_gathered * V_gathered * W_gathered, axis=3)
        
        return values - 1.0

    def objective(x):
        """
        Objective function for least_squares.
        x: flattened vector of [U, V, W]
        Returns: flattened residual vector
        """
        R = x.shape[0] // (n*m + m*p + p*n)
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        res = compute_residuals(U, V, W)
        return res.flatten()

    def try_rank(R, max_restarts=1000):
        """
        Try to find a decomposition of rank R using random restarts and least_squares.
        """
        total_vars = (n*m + m*p + p*n) * R
        best_res_norm = float('inf')
        best_x = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale matters. Small values are often better for starting local optimization.
            scale = 0.1
            x0 = np.random.randn(total_vars) * scale
            
            try:
                res = least_squares(
                    objective, 
                    x0, 
                    method='trf',  # Trust Region Reflective, handles bounds if needed, robust
                    ftol=1e-15, 
                    xtol=1e-15, 
                    gtol=1e-15,
                    max_nfev=1000  # Limit function evaluations per run to avoid hanging
                )
                
                res_norm = np.linalg.norm(res.fun)
                
                if res_norm < best_res_norm:
                    best_res_norm = res_norm
                    best_x = res.x.copy()
                    
                # Check for convergence to exact solution
                if res_norm < 1e-10:
                    # Reconstruct U, V, W
                    U = best_x[:n*m*R].reshape(n*m, R)
                    V = best_x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W = best_x[n*m*R + m*p*R :].reshape(p*n, R)
                    return U, V, W
                    
            except Exception as e:
                # If optimization fails, try next restart
                continue
                
        if best_x is not None and best_res_norm < 1e-8:
            U = best_x[:n*m*R].reshape(n*m, R)
            V = best_x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = best_x[n*m*R + m*p*R :].reshape(p*n, R)
            return U, V, W
            
        return None

    # Known lower bound/rank for 2x5x5 is 38.
    # Schoolbook rank is 50.
    # We try ranks from 38 upwards.
    
    # If we don't find 38, we might find 39, 40...
    # The score is the rank. Lower is better.
    
    # Start with schoolbook as a fallback
    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Search for minimal rank
    # 38 is the theoretical minimum.
    for R in range(38, 51):
        if time.time() > deadline - 20:
            break
            
        # Adjust number of restarts based on time remaining
        time_remaining = deadline - time.time() - 20
        if time_remaining <= 0:
            break
            
        # Estimate time per restart. With vectorized objective, it's very fast.
        # Let's allow up to 2000 restarts if time permits.
        max_restarts = int(min(2000, max(100, time_remaining * 50)))
        
        result = try_rank(R, max_restarts=max_restarts)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = R
            break # Found minimal rank, stop searching higher ranks
            
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
