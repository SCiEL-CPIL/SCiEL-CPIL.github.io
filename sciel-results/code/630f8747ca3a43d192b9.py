import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def score_from_points(points):
        """
        Calculate the score Delta / n^n for a set of points.
        points: (n, 2) array
        """
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / D)
        # Mask upper triangle to get unique pairs
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def log_score_from_points(points):
        """
        Calculate log(Delta) for points assumed to have diameter <= 2.
        This is used for optimization where we enforce diameter <= 2.
        """
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return np.sum(np.log(dists[mask]))

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    best_score = score_from_points(regular_octagon())
    best_points = regular_octagon().copy()

    # --- Phase 1: Optimize angles on a unit circle ---
    # Variables: alpha_1 ... alpha_7, alpha_8 = 2*pi - sum(alpha_1...7)
    
    def cyclic_objective(alpha_vars):
        alphas = np.zeros(n)
        alphas[:7] = alpha_vars
        alphas[7] = 2 * np.pi - np.sum(alpha_vars)
        
        if np.any(alphas <= 1e-9):
            return 1e10
        
        psi = np.cumsum(alphas)
        psi[0] = 0
        
        pts = np.column_stack([np.cos(psi), np.sin(psi)])
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        unique_dists = dists[mask]
        
        # Maximize sum(log(d))
        return -np.sum(np.log(np.maximum(unique_dists, 1e-12)))

    bounds = [(1e-4, 2 * np.pi)] * 7
    
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(
                cyclic_objective,
                bounds=bounds,
                maxiter=300,
                tol=1e-8,
                seed=42,
                popsize=25,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True,
                workers=1
            )
            
            if de_res.fun < 1e9:
                alpha_vars = de_res.x
                alphas = np.zeros(n)
                alphas[:7] = alpha_vars
                alphas[7] = 2 * np.pi - np.sum(alpha_vars)
                
                if np.all(alphas > 1e-9):
                    psi = np.cumsum(alphas)
                    psi[0] = 0
                    pts_unit = np.column_stack([np.cos(psi), np.sin(psi)])
                    
                    # Check score
                    s = score_from_points(pts_unit)
                    if s > best_score:
                        best_score = s
                        best_points = pts_unit.copy()
                        
        except Exception:
            pass

    # --- Phase 2: Refine in full coordinate space ---
    # Start from the best point found so far (likely on the circle)
    # Allow coordinates to move, constrained by max distance <= 2.
    
    def coord_objective(coords):
        points = coords.reshape(n, 2)
        # Objective: Maximize log product of distances
        # Minimize negative log product
        return -log_score_from_points(points)

    def make_constraint(i, j):
        def constraint(coords):
            p1 = coords[2*i:2*i+2]
            p2 = coords[2*j:2*j+2]
            d2 = np.sum((p1 - p2)**2)
            return 4.0 - d2 # Must be >= 0
        return constraint

    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Bounds for coordinates: within [-2, 2] is safe, but tighter is better.
    # Since diameter is 2, points must be in a disk of diameter 2?
    # Not necessarily, but they are within a set of diameter 2.
    # The convex hull is within a circle of diameter 2? No, Jung's theorem says
    # any set of diameter 2 is contained in a circle of radius 2/sqrt(3).
    # So coordinates should be within [-2, 2] roughly.
    bounds_coord = [(-2.0, 2.0)] * (2 * n)

    # Multi-start local refinement
    # We have best_points. Let's try to optimize from there.
    # We might also try a few random perturbations if time allows.
    
    initial_points = best_points.copy()
    # Center the points at origin to help optimization
    centroid = np.mean(initial_points, axis=0)
    initial_points = initial_points - centroid
    
    # Ensure diameter is <= 2 for the start
    D_init = np.max(np.linalg.norm(initial_points[:, np.newaxis, :] - initial_points[np.newaxis, :, :], axis=2))
    if D_init > 2.0:
        initial_points = initial_points * (2.0 / D_init)
        
    x0 = initial_points.flatten()
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                coord_objective,
                x0,
                method='SLSQP',
                bounds=bounds_coord,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success:
                pts_opt = res.x.reshape(n, 2)
                s = score_from_points(pts_opt)
                if s > best_score:
                    best_score = s
                    best_points = pts_opt.copy()
        except Exception:
            pass

    # Try a few random restarts if time permits
    # Generate random points in a disk of radius 1, check diameter, scale.
    # This might find a better local minimum if the circle solution was a local one.
    
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(5): # Up to 5 restarts
            if time.time() > deadline - 5:
                break
            # Random points in unit disk
            r = np.random.rand(n) ** 0.5
            theta = np.random.rand(n) * 2 * np.pi
            pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            # Scale to diameter 2
            D = np.max(np.linalg.norm(pts_rand[:, np.newaxis, :] - pts_rand[np.newaxis, :, :], axis=2))
            if D < 1e-6:
                continue
            pts_scaled = pts_rand * (2.0 / D)
            
            # Center
            pts_scaled = pts_scaled - np.mean(pts_scaled, axis=0)
            
            x0_rand = pts_scaled.flatten()
            
            try:
                res = minimize(
                    coord_objective,
                    x0_rand,
                    method='SLSQP',
                    bounds=bounds_coord,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                if res.success:
                    pts_opt = res.x.reshape(n, 2)
                    s = score_from_points(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_points = pts_opt.copy()
            except Exception:
                pass

    return best_points.tolist()
