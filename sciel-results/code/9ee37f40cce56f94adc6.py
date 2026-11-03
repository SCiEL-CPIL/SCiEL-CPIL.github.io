import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
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

    def _compute_areas(pts):
        """Compute all triangle areas for given points."""
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        return 0.5 * cross

    def _penalty_objective(points_flat, t_target, mu=100.0):
        """
        Penalty-based objective: minimize -t + mu * sum(max(0, t - area_i)^2)
        This encourages all triangle areas to be >= t_target.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        areas = _compute_areas(pts)
        
        # Penalty for triangles with area < t_target
        violations = np.maximum(0, t_target - areas)
        penalty = mu * np.sum(violations ** 2)
        
        # We want to maximize t, so minimize -t
        return -t_target + penalty

    def _soft_min_objective(points_flat, beta=100.0):
        """
        Soft-min objective using log-sum-exp for smoothness.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        areas = _compute_areas(pts)
        
        # Soft min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Hard objective for final polish.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _optimize_with_penalty_method(init_points, max_time=30.0):
        """
        Penalty-based max-min optimization:
        1. Start with a small t_target
        2. Optimize points to satisfy area_i >= t_target
        3. Increase t_target and repeat
        4. Return the best configuration found
        """
        current = init_points.copy()
        best_score = -float('inf')
        best_points = current.copy()
        start = time.time()
        
        # Get initial score
        init_score = compute_score(current)
        
        # Start t_target slightly below current min area
        pts_norm = _normalize_points(current)
        areas = _compute_areas(pts_norm)
        min_area = np.min(areas)
        hull_area = 1.0  # normalized
        t_target = 0.8 * min_area
        
        # Penalty parameter
        mu = 50.0
        
        # Iterate: increase t_target, optimize points
        max_iterations = 20
        for iteration in range(max_iterations):
            if time.time() - start > max_time - 5:
                break
            
            # Optimize points to satisfy constraints
            try:
                res = opt.minimize(
                    lambda x: _penalty_objective(x, t_target, mu),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
            
            # Check if constraints are satisfied
            pts_norm = _normalize_points(current)
            areas = _compute_areas(pts_norm)
            min_area_now = np.min(areas)
            
            # Update best
            score = compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
            
            # If all areas >= t_target, increase t_target
            if min_area_now >= t_target * 0.95:
                # Increase t_target
                t_target = min_area_now * 1.05
                # Increase penalty
                mu = min(mu * 1.5, 500.0)
            else:
                # If stuck, try to increase t_target slightly and increase penalty
                if mu < 200.0:
                    mu = min(mu * 1.5, 500.0)
                else:
                    # Can't increase penalty much, try small increase in t
                    t_target = min_area_now * 1.02
                    if t_target <= t_target * 0.99:  # No progress
                        break
        
        # Final polish with soft-min
        if time.time() - start < max_time - 3:
            try:
                res = opt.minimize(
                    lambda x: _soft_min_objective(x, beta=150.0),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
            
            score = compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        
        # Final hard polish
        if time.time() - start < max_time - 1:
            try:
                res = opt.minimize(
                    _hard_objective,
                    current.flatten(),
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
            except Exception:
                pass
        
        return best_points, best_score

    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 1 center + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
        
    # 9. 5 outer + 8 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 10. 4 outer + 9 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r)
        initializations.append(np.vstack([outer, inner]))
        
    # Add some random perturbations to top symmetric configs to break symmetry if beneficial
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization with penalty method
    for init in initializations:
        if time.time() > start_time + time_limit - 15:
            break
        
        try:
            candidate, score = _optimize_with_penalty_method(init, max_time=25.0)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
