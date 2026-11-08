import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_min(areas, beta):
        """Compute soft minimum of areas using log-sum-exp trick."""
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        return -log_sum_exp / beta

    def _soft_objective(points_flat, beta=100.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _refine_coordinate_descent(points, max_iter=200, tol=1e-10):
        """Coordinate descent refinement for the hard objective."""
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        
        # Adaptive step sizes
        base_steps = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
            improved = False
            for i in range(N):
                # Try perturbations with decreasing step sizes
                for scale in base_steps:
                    for dx in [-scale, scale]:
                        for dy in [-scale, scale]:
                            trial = pts.copy()
                            trial[i, 0] += dx
                            trial[i, 1] += dy
                            trial_norm = _normalize_points(trial)
                            s = compute_score(trial_norm)
                            if s > best_score + tol:
                                pts = trial_norm
                                best_score = s
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return pts, best_score

    def _refine_powell(points):
        """Powell refinement for the hard objective."""
        try:
            res = opt.minimize(
                _hard_objective,
                points.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return points, compute_score(points)

    # Generate diverse initial configurations - focus on most promising
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 7 outer + 6 inner (two rings) - most promising structure
    for r_inner in [0.35, 0.45, 0.55, 0.65]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        initializations.append(np.vstack([outer, inner]))
    
    # 4. 6 outer + 6 inner + 1 center
    for r_inner in [0.35, 0.45, 0.55, 0.65]:
        outer = _regular_polygon(6, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        center = np.array([[0.0, 0.0]])
        initializations.append(np.vstack([outer, inner, center]))
    
    # 5. 8 outer + 5 inner
    for r_inner in [0.35, 0.5, 0.65]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 6. 9 outer + 4 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 7. 10 outer + 3 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 8. 11 outer + 2 inner
    for r_inner in [0.4, 0.55]:
        outer = _regular_polygon(11, 1.0)
        inner = _regular_polygon(2, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 9. 5 outer + 5 middle + 3 inner
    for r_mid, r_inner in [(0.6, 0.3), (0.5, 0.25), (0.7, 0.4)]:
        outer = _regular_polygon(5, 1.0)
        middle = _regular_polygon(5, r_mid, offset=np.pi/5)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, middle, inner]))
    
    # 10. Random perturbations of top configs
    rng = np.random.RandomState(42)
    for i in range(0, min(8, len(initializations)), 2):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)
    
    # 11. Additional random starts with structured patterns
    for _ in range(5):
        n_outer = rng.choice([6, 7, 8, 9])
        n_inner = N - n_outer
        r_outer = rng.uniform(0.9, 1.0)
        r_inner = rng.uniform(0.35, 0.6)
        outer = _regular_polygon(n_outer, r_outer)
        inner = _regular_polygon(n_inner, r_inner, offset=rng.uniform(0, 2*np.pi/max(n_inner,1)))
        pts = np.vstack([outer, inner])
        pts += rng.randn(*pts.shape) * 0.05
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    n_inits = len(initializations)
    
    # Optimize each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 8:
            break
            
        current = _normalize_points(init.copy())
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 5:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Coordinate descent refinement
        if time.time() > start_time + time_limit - 3:
            break
        current, score_cd = _refine_coordinate_descent(current, max_iter=100)
        
        # Step 3: Powell polish
        if time.time() > start_time + time_limit - 1:
            break
        current, score_pw = _refine_powell(current)
        
        score = max(score_cd, score_pw)
        
        if score > best_score:
            best_score = score
            best_points = current.copy()

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = _normalize_points(initializations[0])

    return best_points
