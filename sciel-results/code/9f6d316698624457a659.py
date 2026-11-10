import numpy as np
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Initial guess: points on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also try a grid-like initialization
    # 4x4 grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    points1 = np.column_stack([gx.ravel(), gy.ravel()])
    
    # We will run multiple optimizations and pick the best
    best_points = points0
    best_score = -np.inf
    
    beta = 10.0 # Sharpness of the soft min/max
    
    def logsumexp(x):
        m = np.max(x)
        if m > 500: # Avoid overflow
            return m + np.log(np.sum(np.exp(x - m)))
        return m + np.log(np.sum(np.exp(x - m)))

    def objective(x):
        pts = x.reshape(n, d)
        # Compute pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff**2, axis=2)
        # Get upper triangle
        iu = np.triu_indices(n, k=1)
        d2 = dist2[iu]
        
        # Soft min and max
        # log(min) approx -1/beta * log(sum(exp(-beta * d2)))
        # log(max) approx 1/beta * log(sum(exp(beta * d2)))
        # We want to minimize log(max) - log(min)
        # = 1/beta * (log(sum(exp(beta * d2))) + log(sum(exp(-beta * d2))))
        
        term1 = logsumexp(beta * d2)
        term2 = logsumexp(-beta * d2)
        return (term1 + term2) / beta

    def gradient(x):
        # Numerical gradient is safer and fast enough for 32 params
        # But L-BFGS-B benefits from analytic gradient.
        # Let's stick to finite difference for robustness or use Powell.
        # Powell is derivative-free and robust.
        pass

    # Let's use Powell for robustness, or L-BFGS-B with finite diff
    # Powell might be slower but more robust to non-smoothness (though we smoothed it)
    # Let's try L-BFGS-B with finite difference gradient
    # 32 parameters, 600s is plenty.
    
    methods = ['Powell', 'L-BFGS-B']
    inits = [points0, points1]
    
    for init in inits:
        x0 = init.ravel()
        for method in methods:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, x0, method=method, options={'maxiter': 1000, 'maxfev': 10000})
                if res.success or res.fun < best_score:
                    # Evaluate true score
                    pts = res.x.reshape(n, d)
                    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dist = np.sqrt(np.sum(diff**2, axis=2))
                    iu = np.triu_indices(n, k=1)
                    d_true = dist[iu]
                    d_min = np.min(d_true)
                    d_max = np.max(d_true)
                    if d_max > 0:
                        score = (d_min / d_max)**2
                    else:
                        score = 0
                    if score > best_score:
                        best_score = score
                        best_points = pts
            except Exception:
                pass
                
    # One more refinement with exact objective using Nelder-Mead?
    # Or just return best_points
    
    return best_points
