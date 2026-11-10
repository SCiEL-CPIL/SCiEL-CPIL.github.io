import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Precompute indices for projections
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y, indexing='xy') # yy is y, xx is x
    # P[y, x] corresponds to yy[y, x] and xx[y, x]
    
    # Projection values
    proj_m1 = xx - yy      # x - y
    proj_0 = xx            # x
    proj_1 = xx + yy       # x + y
    proj_2 = xx + 2 * yy   # x + 2y
    
    # Unique values and inverse indices for binning
    # We need to sum probabilities for each unique value
    
    def get_distribution(projection, P):
        # Flatten projection and P
        # proj is 2D, P is 2D
        # We need to sum P[y,x] for all (y,x) where projection[y,x] == val
        # Using np.unique with return_inverse
        vals, inverse = np.unique(projection, return_inverse=True)
        # Sum P along the inverse
        # P_flat is 1D
        P_flat = P.flatten()
        dist = np.bincount(inverse.flatten(), weights=P_flat.flatten(), minlength=len(vals))
        # Filter zeros for entropy calculation
        dist = dist[dist > 1e-15]
        if len(dist) == 0:
            return 0.0
        # Normalize? P is already normalized. But bincount sums to sum(P)=1.
        # So dist is a probability distribution.
        # Calculate Shannon entropy
        return -np.sum(dist * np.log2(dist))
    
    def compute_entropies(P):
        H_m1 = get_distribution(proj_m1, P)
        H_0 = get_distribution(proj_0, P)
        H_1 = get_distribution(proj_1, P)
        H_2 = get_distribution(proj_2, P)
        return H_m1, H_0, H_1, H_2
    
    def objective(p_flat):
        P = p_flat.reshape((n, n))
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        if H_m1 < 1e-9:
            return 0.0
        # Soft max for M
        t = 20.0
        Hs = np.array([H_0, H_1, H_2])
        M_soft = (1.0 / t) * np.log(np.sum(np.exp(t * Hs)))
        # Maximize ratio D/M => Minimize -log(D/M) = log(M) - log(D)
        # Or minimize -D/M
        # Using log for stability
        if M_soft < 1e-9:
            return 0.0
        return -np.log(H_m1) + np.log(M_soft)
    
    # Initial guess
    p0 = P0.flatten()
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    bounds = [(0, 1)] * (n * n)
    
    start_time = time.time()
    best_p = p0
    best_score = -objective(p0) # We want to maximize score, objective minimizes negative log ratio
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B is good for bounded constraints
    # But gradients are not provided. It will approximate.
    # Given 100 vars, it might be slow if not careful.
    # Let's try a few iterations.
    
    try:
        res = minimize(
            objective, 
            p0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        if res.success:
            best_p = res.x
            best_score = -res.fun
            
        # Check time
        if time.time() - start_time > 600:
            pass # Just break, but minimize handles its own loop
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        best_p = p0
    
    # Convert back to 2D
    P_best = best_p.reshape((n, n))
    
    # Final validation and cleanup
    P_best = np.clip(P_best, 0, None)
    s = np.sum(P_best)
    if s > 0:
        P_best /= s
    else:
        P_best = P0
        
    # Ensure finite
    if not np.all(np.isfinite(P_best)):
        P_best = P0
        
    return P_best
