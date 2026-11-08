"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0


def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Use a radial function r(theta) = 1 + sum of Fourier terms to define
    a convex region, place points on the boundary, then optimize.
    Also try regular polygon and other structured initializations.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Try several structured initializations
    initializations = []
    
    # 1. Regular 13-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    initializations.append(points_reg)
    
    # 2. Points on a circle with slight perturbation
    rng = np.random.RandomState(42)
    angles_pert = angles_reg + 0.1 * rng.randn(N)
    angles_pert = np.sort(angles_pert)
    points_pert = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)])
    initializations.append(points_pert)
    
    # 3. Random points in unit disk
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.RandomState(seed)
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        points_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(points_rand)
    
    # 4. Points on an ellipse
    for a, b in [(1.0, 0.8), (1.2, 0.9), (0.9, 1.1)]:
        angles_ell = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points_ell = np.column_stack([a * np.cos(angles_ell), b * np.sin(angles_ell)])
        initializations.append(points_ell)
    
    # Evaluate all initializations
    for pts in initializations:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Use differential evolution to optimize the 2D coordinates
    # We'll optimize in a bounded box, then normalize
    def objective(x):
        """Negative score for minimization."""
        pts = x.reshape(N, 2)
        # Normalize to unit area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-12:
                return 0.0
            # Normalize
            pts_norm = pts / np.sqrt(hull_area)
            min_area = _min_triangle_area(pts_norm)
            score = min_area  # Since hull area is now 1
            return -score
        except Exception:
            return 0.0
    
    # Run differential evolution with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initial populations
        if restart == 0:
            # Start from best found so far
            bounds = [(pts - 0.1, pts + 0.1) for pts in best_points.flatten()]
        else:
            # Random bounds around origin
            bounds = [(-1.5, 1.5)] * (2 * N)
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=restart * 100,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=True
            )
            
            if result.success or result.fun < 0:
                pts_opt = result.x.reshape(N, 2)
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_points = pts_opt.copy()
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B on the best solution
    if best_points is not None and time.time() < deadline:
        def local_objective(x):
            pts = x.reshape(N, 2)
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area <= 1e-12:
                    return 0.0
                pts_norm = pts / np.sqrt(hull_area)
                min_area = _min_triangle_area(pts_norm)
                return -min_area
            except Exception:
                return 0.0
        
        try:
            result = minimize(
                local_objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts_refined = result.x.reshape(N, 2)
            s = _score(pts_refined)
            if s > best_score:
                best_score = s
                best_points = pts_refined.copy()
        except Exception:
            pass
    
    # If we still have time, try more differential evolution with tighter bounds
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline:
                break
            bounds = [(p - 0.05, p + 0.05) for p in best_points.flatten()]
            try:
                result = differential_evolution(
                    objective,
                    bounds=bounds,
                    seed=int(time.time() * 1000) % 10000,
                    maxiter=100,
                    popsize=10,
                    tol=1e-10,
                    polish=True
                )
                if result.fun < 0:
                    pts_opt = result.x.reshape(N, 2)
                    s = _score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_points = pts_opt.copy()
            except Exception:
                pass
    
    # Final check
    if best_points is None:
        # Fallback to regular polygon
        angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    
    return best_points
# EVOLVE-BLOCK-END
