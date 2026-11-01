import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing. Area per circle ~ pi * r^2. 
    # Total area pi * 1^2. 
    # N * pi * r^2 <= pi => r <= 1/sqrt(N) ~ 0.057.
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069.
    # Effective area per circle is 1/N.
    # Spacing s ~ sqrt(1/N / 0.9) ~ sqrt(3.3) ~ 1.8? No.
    # Area of hexagon with side s/2 is (3*sqrt(3)/2) * (s/2)^2 = (3*sqrt(3)/8) s^2.
    # N * (3*sqrt(3)/8) s^2 = pi.
    # s^2 = 8*pi / (3*sqrt(3)*N).
    # s = sqrt(8*pi / (3*sqrt(3)*300)) ~ sqrt(25.13 / 1558) ~ sqrt(0.016) ~ 0.126.
    
    s = np.sqrt(8 * np.pi / (3 * np.sqrt(3) * N))
    
    # Generate hex grid
    pts = []
    # Range of indices
    # Max radius is 1.
    # Points (i, j) in hex grid.
    # x = s * (i + 0.5 * (j % 2))
    # y = s * (sqrt(3)/2 * j)
    
    # We need to generate enough points and then filter.
    # Max i, j such that distance <= 1.
    # s * sqrt(i^2 + i*j + j^2) <= 1 (approx)
    # Let's just iterate a square range and filter.
    max_idx = int(1.0 / s) + 2
    
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * (np.sqrt(3) / 2 * j)
            if np.hypot(x, y) <= 1.0:
                pts.append([x, y])
                
    pts = np.array(pts)
    
    # If we have more than N points, select the N closest to center? 
    # Or just take the first N? 
    # Taking closest to center is usually better for packing.
    if len(pts) > N:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Add random points if not enough
        while len(pts) < N:
            x, y = np.random.uniform(-1, 1, 2)
            if np.hypot(x, y) <= 1.0:
                pts = np.vstack([pts, [x, y]])
    
    pts = pts[:N]
    
    # 2. Optimization
    # Objective: minimize -softmin( pairwise_dists/2, boundary_dists )
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        p = params.reshape(-1, 2)
        
        # Boundary distances
        r_pts = np.linalg.norm(p, axis=1)
        boundary_dists = 1.0 - r_pts
        
        # Pairwise distances
        # pdist returns condensed vector of length N*(N-1)/2
        d = pdist(p)
        pair_dists = d / 2.0
        
        # Combine
        all_dists = np.concatenate([pair_dists, boundary_dists])
        
        # Softmin
        # To avoid overflow/underflow in exp, subtract max
        # softmin(x) = -tau * log(sum(exp(-x/tau)))
        # = -tau * log(sum(exp(-(x - max_x)/tau))) - max_x
        # Let's use a stable version.
        tau = 0.01
        # We want to minimize -softmin.
        # softmin is approx min.
        # If min is large, -softmin is small (negative).
        # We want to maximize min, so minimize -min.
        
        # Stable log-sum-exp for -x/tau
        # Let y = -x/tau. We want log(sum(exp(y))).
        # y_max = max(y) = -min(x)/tau.
        # log(sum(exp(y))) = y_max + log(sum(exp(y - y_max)))
        # softmin = -tau * (y_max + log(sum(exp(y - y_max))))
        #          = min(x) + tau * log(sum(exp(-(x - min(x))/tau)))
        # This is stable.
        
        min_d = np.min(all_dists)
        # If min_d is very negative, exp might underflow, but that's fine.
        # We subtract min_d to make arguments <= 0.
        args = -(all_dists - min_d) / tau
        # args are <= 0. exp(args) <= 1.
        log_sum = np.log(np.sum(np.exp(args)))
        soft_min_val = min_d + tau * log_sum
        
        return -soft_min_val
    
    # Initial guess
    x0 = pts.flatten()
    
    # Bounds
    # Points must be within unit circle, but we allow slightly outside to let optimizer fix it.
    # Actually, if a point is outside, boundary_dist is negative.
    # The softmin will be negative. -softmin will be positive.
    # We want to minimize this.
    # So it will push points inside.
    bounds = [(-1.1, 1.1)] * (2 * N)
    
    # Optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.fun < 1e9:
            pts_opt = res.x.reshape(-1, 2)
            # Check if valid
            if np.all(np.isfinite(pts_opt)):
                pts = pts_opt
    except Exception:
        pass
    
    # Final check and return
    # Ensure points are strictly inside? 
    # The score is min(min_pair/2, min_boundary).
    # If a point is outside, min_boundary < 0, score < 0.
    # We want to maximize score.
    # The optimizer should have pushed them inside.
    
    return pts
