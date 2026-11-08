import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

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

    # Generate diverse initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 7 outer + 6 inner (two rings)
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)  # offset by half step
        initializations.append(np.vstack([outer, inner]))
    
    # 4. 6 outer + 6 inner + 1 center
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        outer = _regular_polygon(6, 1.0)
        inner = _regular_polygon(6, r_inner, offset=np.pi/6)
        center = np.array([[0.0, 0.0]])
        initializations.append(np.vstack([outer, inner, center]))
    
    # 5. 5 outer + 5 middle + 3 inner
    for r_mid, r_inner in [(0.6, 0.3), (0.5, 0.25), (0.7, 0.4), (0.65, 0.35)]:
        outer = _regular_polygon(5, 1.0)
        middle = _regular_polygon(5, r_mid, offset=np.pi/5)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, middle, inner]))
    
    # 6. 4 outer + 4 middle + 4 inner + 1 center
    for r_mid, r_inner in [(0.7, 0.4), (0.6, 0.35), (0.5, 0.3)]:
        outer = _regular_polygon(4, 1.0)
        middle = _regular_polygon(4, r_mid, offset=np.pi/4)
        inner = _regular_polygon(4, r_inner)
        center = np.array([[0.0, 0.0]])
        initializations.append(np.vstack([outer, middle, inner, center]))
    
    # 7. 8 outer + 5 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 8. 9 outer + 4 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 9. 10 outer + 3 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 10. 11 outer + 2 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(11, 1.0)
        inner = _regular_polygon(2, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 11. 3 outer + 10 inner (triangle outer)
    for r_inner in [0.4, 0.5, 0.6]:
        outer = _regular_polygon(3, 1.0)
        inner = _regular_polygon(10, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 12. 2 outer + 11 inner
    for r_inner in [0.5, 0.6, 0.7]:
        outer = _regular_polygon(2, 1.0)
        inner = _regular_polygon(11, r_inner)
        initializations.append(np.vstack([outer, inner]))
    
    # 13. Random perturbations of symmetric configs
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 2):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.15
        initializations.append(perturbed)
    
    # 14. More random starts
    for _ in range(10):
        r_outer = np.random.uniform(0.8, 1.0)
        r_inner = np.random.uniform(0.3, 0.6)
        n_outer = np.random.choice([5, 6, 7, 8, 9])
        n_inner = N - n_outer
        outer = _regular_polygon(n_outer, r_outer)
        if n_inner > 0:
            inner = _regular_polygon(n_inner, r_inner, offset=np.random.uniform(0, 2*np.pi/n_inner))
            pts = np.vstack([outer, inner])
        else:
            pts = outer
        pts += rng.randn(*pts.shape) * 0.1
        initializations.append(pts)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > start_time + time_limit - 15:
            break
            
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [30.0, 80.0, 150.0, 300.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 8:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Powell with hard objective (robust local search)
        if time.time() > start_time + time_limit - 3:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead as alternative
        if time.time() > start_time + time_limit - 1:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
