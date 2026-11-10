import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    
    # Objective function to minimize (negative log score)
    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        # Avoid log(0) and ensure distinctness
        # If any distance is too small, return a large penalty
        # We want to maximize log(product of distances) - 56 * log(diameter)
        # Let's add a penalty for min distance < 1e-6
        
        min_dist = np.min(dists[~np.eye(n, dtype=bool)])
        if min_dist < 1e-8:
            return 1e10 + 1e10 / (min_dist + 1e-15)
            
        # Set diagonal to 1 to ignore self-distance for log sum
        # But we need to be careful with the diameter calculation
        # Diameter is max distance
        D = np.max(dists)
        if D < 1e-9:
            return 1e10
            
        # Log product of all ordered pairs (i != j)
        # Sum of log(dists) for i != j
        # We can use the full matrix sum and subtract diagonal (log(1)=0 if we set them to 1)
        # But dists has 0 on diagonal. 
        # Let's just sum off-diagonal elements.
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diagonal_dists = dists[mask]
        
        log_prod = np.sum(np.log(off_diagonal_dists))
        
        # Score term: log_prod - 56 * log(D)
        # We minimize the negative
        return -(log_prod - 56 * np.log(D))

    best_score = -np.inf
    best_points = None
    
    # Run multiple optimizations from different starting points
    # 1. Regular octagon
    # 2. Several random points
    
    def run_optimization(x0):
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'xtol': 1e-12})
            if res.fun < best_score:
                return res.x.reshape(n, 2), res.fun
        except Exception:
            pass
        return None, -np.inf

    # 1. Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_oct = np.column_stack((np.cos(angles), np.sin(angles)))
    # Scale to diameter 2 (radius 1 is diameter 2 for regular polygon? No, distance between opposite points is 2R. So R=1 gives D=2)
    # The objective handles scaling internally via the D term, so initial scale doesn't matter much, 
    # but it helps to be in a reasonable range.
    p, s = run_optimization(points_oct.flatten())
    if p is not None:
        best_points = p
        best_score = s

    # 2. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Generate random points in a unit square or circle
        # Let's use random points in [-1, 1] x [-1, 1]
        pts_rand = rng.uniform(-1, 1, (n, 2))
        # Ensure they are distinct enough
        p, s = run_optimization(pts_rand.flatten())
        if p is not None and s < best_score:
            best_points = p
            best_score = s

    if best_points is None:
        # Fallback to regular octagon
        return [[float(p[0]), float(p[1])] for p in points_oct]

    return [[float(p[0]), float(p[1])] for p in best_points]
