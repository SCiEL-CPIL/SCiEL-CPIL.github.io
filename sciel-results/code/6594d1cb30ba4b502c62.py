import numpy as np
from scipy.optimize import minimize

def construct():
    n = 16
    d = 2
    
    def objective(params):
        # Reshape params to (n, d)
        pts = params.reshape(n, d)
        
        # Translate to origin
        pts = pts - np.mean(pts, axis=0)
        
        # Scale to unit norm (sum of squared norms = 1)
        norm_sq = np.sum(pts**2)
        if norm_sq < 1e-12:
            return 1e12 # Avoid division by zero
        pts = pts / np.sqrt(norm_sq)
        
        # Compute pairwise squared distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        max_dist_sq = np.max(dist_sq)
        
        if min_dist_sq < 1e-12:
            return 1e12
            
        return max_dist_sq / min_dist_sq

    best_score = float('inf')
    best_params = None
    
    # Multi-start
    n_starts = 50
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        # Generate initial guess
        # Mix of random and structured
        if i % 2 == 0:
            # Random
            x0 = rng.uniform(-1, 1, size=(n, d)).flatten()
        else:
            # Structured: Circle with noise
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.uniform(-0.1, 0.1, n)
            radii = 1.0 + rng.uniform(-0.2, 0.2, n)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            
        # Optimize
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Reconstruct points from best_params
    # The objective function normalizes, so we need to return the points in a valid form.
    # The score is scale invariant, so we can return the normalized points or scaled up.
    # Let's return the points from the best optimization result, normalized as in objective.
    pts = best_params.reshape(n, d)
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq > 1e-12:
        pts = pts / np.sqrt(norm_sq)
        
    return pts
