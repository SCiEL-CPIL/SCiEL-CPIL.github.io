import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n = 8
    # Regular octagon as fallback
    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def score(points):
        # points: (n, 2)
        # Scale to diameter 2
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        # Product of distances for i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    start_time = time.time()
    timeout = 140  # seconds, keep 10s margin
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    # Parametric optimization:
    # Fix p0 = (-1, 0), p1 = (1, 0) as the diameter pair.
    # Optimize p2..p7.
    # Constraints: d(pi, pj) <= 2 for all i,j.
    
    def objective(x):
        # x: (12,) representing p2..p7
        pts = np.zeros((n, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n-2, 2)
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Check validity (distinct points)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.min(dists[mask]) < 1e-6:
            return 1e10
            
        # Log sum of distances
        log_sum = np.sum(np.log(dists[mask]))
        return -log_sum

    # Constraints for SLSQP: g(x) >= 0
    # We want d(pi, pj) <= 2, so 2 - d(pi, pj) >= 0
    def make_constraint(i, j):
        def constraint(x):
            pts = np.zeros((n, 2))
            pts[0] = [-1.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = x.reshape(n-2, 2)
            d = np.linalg.norm(pts[i] - pts[j])
            return 2.0 - d
        return constraint

    constraints = []
    # Pairs among 2..7
    for i in range(2, n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    # Pairs between 0 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(0, i)})
    # Pairs between 1 and 2..7
    for i in range(2, n):
        constraints.append({'type': 'ineq', 'fun': make_constraint(1, i)})
    
    # Bounds for x (coordinates of p2..p7)
    # x in [-1, 1], y in [-1.8, 1.8]
    # This is tighter than [-3, 3] and safe for the lens.
    bounds = [(-1.0, 1.0), (-1.8, 1.8)] * (n-2)
    
    # Initial guesses
    # 1. Regular octagon perturbed to match p0, p1
    oct_pts = regular_octagon()
    # Indices: 0:(1,0), 1:(.707,.707), 2:(0,1), 3:(-.707,.707), 4:(-1,0), 5:(-.707,-.707), 6:(0,-1), 7:(.707,-.707)
    # p0=(-1,0) is idx 4. p1=(1,0) is idx 0.
    # Remaining: 1,2,3,5,6,7
    init_x = np.array([
        oct_pts[1][0], oct_pts[1][1],
        oct_pts[2][0], oct_pts[2][1],
        oct_pts[3][0], oct_pts[3][1],
        oct_pts[5][0], oct_pts[5][1],
        oct_pts[6][0], oct_pts[6][1],
        oct_pts[7][0], oct_pts[7][1]
    ])
    
    def run_slsqp(x0, maxiter, ftol):
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success:
                pts = np.zeros((n, 2))
                pts[0] = [-1.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = res.x.reshape(n-2, 2)
                
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                D = np.max(dists)
                
                if D <= 2.0 + 1e-5:
                    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                    if np.min(dists[mask]) > 1e-5:
                        return pts
            return None
        except Exception:
            return None

    # Run 1: From Octagon
    if time.time() - start_time < timeout:
        pts = run_slsqp(init_x, 500, 1e-10)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    # Run 2..N: Random restarts
    if time.time() - start_time < timeout:
        np.random.seed(42)
        for k in range(20):
            if time.time() - start_time >= timeout:
                break
            pts_init = np.zeros((n, 2))
            pts_init[0] = [-1.0, 0.0]
            pts_init[1] = [1.0, 0.0]
            for i in range(2, n):
                # Sample x in [-1, 1]
                x = np.random.uniform(-1.0, 1.0)
                # Max y^2 is min(4-(x-1)^2, 4-(x+1)^2)
                # 4-(x-1)^2 = 3 - x^2 + 2x
                # 4-(x+1)^2 = 3 - x^2 - 2x
                # min is 3 - x^2 - 2|x|
                max_y_sq = 3.0 - x*x - 2.0*abs(x)
                if max_y_sq < 0:
                    max_y_sq = 0.0
                max_y = np.sqrt(max_y_sq)
                y = np.random.uniform(-max_y, max_y)
                pts_init[i] = [x, y]
            
            x0 = pts_init[2:].flatten()
            pts = run_slsqp(x0, 300, 1e-10)
            if pts is not None:
                cand_score = score(pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = pts

    # Polish step: Run SLSQP on the best point found with higher precision
    if time.time() - start_time < timeout:
        # Extract x from best_points
        x_best = best_points[2:].flatten()
        pts = run_slsqp(x_best, 1000, 1e-12)
        if pts is not None:
            cand_score = score(pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = pts

    return best_points.tolist()
