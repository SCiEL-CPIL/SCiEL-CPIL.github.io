import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 110.0 

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

    def _fast_soft_min(areas, beta):
        """Fast soft-min using log-sum-exp with clipping for speed."""
        # Clip to avoid overflow and improve numerical stability
        clipped = np.clip(areas, 1e-10, None)
        neg_beta_areas = -beta * clipped
        max_val = np.max(neg_beta_areas)
        if max_val < -500:
            return 1e6
        exps = np.exp(neg_beta_areas - max_val)
        log_sum = max_val + np.log(np.sum(exps))
        return -log_sum / beta

    def _soft_objective(points_flat, beta):
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
        
        soft_min_area = _fast_soft_min(areas, beta)
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
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
        min_area = np.min(areas)
        return -min_area

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), rotation=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _generate_initializations():
        """Generate diverse initial configurations."""
        inits = []
        
        # 1. Regular 13-gon
        inits.append(_regular_polygon(13, 1.0))
        
        # 2. 12 outer + 1 center
        inits.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
        
        # 3. Two concentric rings with different counts
        for n_outer, n_inner, r_inner in [
            (11, 2, 0.3), (11, 2, 0.5), (11, 2, 0.7),
            (10, 3, 0.3), (10, 3, 0.5), (10, 3, 0.7),
            (9, 4, 0.3), (9, 4, 0.5), (9, 4, 0.7),
            (8, 5, 0.3), (8, 5, 0.5), (8, 5, 0.7),
            (7, 6, 0.3), (7, 6, 0.5), (7, 6, 0.7),
            (6, 6, 0.3), (6, 6, 0.5), (6, 6, 0.7),
            (5, 8, 0.3), (5, 8, 0.5), (5, 8, 0.7),
        ]:
            outer = _regular_polygon(n_outer, 1.0)
            inner = _regular_polygon(n_inner, r_inner)
            inits.append(np.vstack([outer, inner]))
        
        # 4. Three concentric rings
        for n1, n2, n3, r1, r2 in [
            (7, 4, 2, 0.4, 0.7),
            (6, 4, 3, 0.4, 0.7),
            (5, 5, 3, 0.4, 0.7),
            (4, 5, 4, 0.4, 0.7),
            (6, 3, 4, 0.3, 0.6),
            (5, 4, 4, 0.3, 0.6),
        ]:
            ring1 = _regular_polygon(n1, 1.0)
            ring2 = _regular_polygon(n2, r1)
            ring3 = _regular_polygon(n3, r2)
            inits.append(np.vstack([ring1, ring2, ring3]))
        
        # 5. Grid-like patterns
        for spacing in [0.3, 0.4, 0.5]:
            grid = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    grid.append([i * spacing, j * spacing])
            grid = np.array(grid[:13])
            inits.append(grid)
        
        # 6. Random but structured patterns
        rng = np.random.RandomState(42)
        for _ in range(10):
            # Mix of random and structured
            pts = []
            # Some on a circle
            n_circle = rng.randint(5, 10)
            angles = np.sort(rng.uniform(0, 2 * np.pi, n_circle))
            for a in angles:
                pts.append([np.cos(a), np.sin(a)])
            # Rest random inside
            while len(pts) < 13:
                r = rng.uniform(0, 1)
                a = rng.uniform(0, 2 * np.pi)
                pts.append([r * np.cos(a), r * np.sin(a)])
            inits.append(np.array(pts))
        
        # 7. Perturbed regular polygons
        for base_idx in [0, 1]:
            base = inits[base_idx]
            for _ in range(3):
                p = base.copy()
                p += rng.randn(*p.shape) * 0.15
                inits.append(p)
        
        return inits

    best_score = -float('inf')
    best_points = None

    initializations = _generate_initializations()

    # Optimization strategy:
    # 1. Soft min annealing with L-BFGS-B
    # 2. Hard min polish with Powell
    
    # Sort initializations by initial score to prioritize good starts
    init_scores = []
    for init in initializations:
        if time.time() > start_time + time_limit - 30:
            break
        s = compute_score(init)
        init_scores.append(s)
    
    # Sort by score descending
    sorted_indices = np.argsort(init_scores)[::-1]
    
    # Only optimize top candidates to save time
    max_inits_to_optimize = min(len(sorted_indices), 20)
    
    for idx in sorted_indices[:max_inits_to_optimize]:
        if time.time() > start_time + time_limit - 15:
            break
            
        current = initializations[idx].copy().flatten()
        
        # Annealing betas - fewer but more strategic
        betas = [20.0, 60.0, 150.0, 400.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 5:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                current = res.x
            except Exception:
                pass
        
        # Final polish with Powell on hard objective
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-15, 'ftol': 1e-15, 'maxfev': 8000}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            # Fallback to the soft-min result
            candidate = current.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        
        # Local search refinement: try small perturbations of the best found so far
        if best_points is not None and time.time() < start_time + time_limit - 3:
            for _ in range(5):
                if time.time() > start_time + time_limit - 3:
                    break
                perturbed = best_points.copy() + np.random.randn(*best_points.shape) * 0.01
                try:
                    res = opt.minimize(
                        _hard_objective,
                        perturbed.flatten(),
                        method='Powell',
                        options={'maxiter': 50, 'xtol': 1e-14, 'ftol': 1e-14, 'maxfev': 2000}
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
