import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    n_triples = len(triples)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0, 0.0
            
        if area_hull < 1e-9:
            return 0.0, 0.0

        # Triangle areas
        min_area = float('inf')
        x = points[:, 0]
        y = points[:, 1]
        for i, j, k in triples:
            v1x = x[j] - x[i]
            v1y = y[j] - y[i]
            v2x = x[k] - x[i]
            v2y = y[k] - y[i]
            cross = abs(v1x * v2y - v1y * v2x)
            area = 0.5 * cross
            if area < min_area:
                min_area = area

        score = min_area / area_hull
        return score, area_hull

    def objective(theta):
        """Objective function for Nelder-Mead on angular parameters."""
        x = np.cos(theta)
        y = np.sin(theta)
        points = np.column_stack((x, y))
        score, _ = compute_score(points)
        return -score

    def objective_flat(points_flat):
        """Objective function for L-BFGS-B on flat coordinates."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_initial_configurations():
        """Create several diverse initial configurations."""
        configs = []
        
        # Config 1: Regular 13-gon (all on hull)
        configs.append(_regular_polygon(13, radius=1.0))
        
        # Config 2: Regular 12-gon with center point
        hull_pts = _regular_polygon(12, radius=1.0)
        center = np.array([[0.0, 0.0]])
        configs.append(np.vstack([hull_pts, center]))
        
        # Config 3: Regular 11-gon + 2 interior points
        hull_pts = _regular_polygon(11, radius=1.0)
        interior1 = np.array([[0.3, 0.0]])
        interior2 = np.array([[-0.15, 0.26]])
        configs.append(np.vstack([hull_pts, interior1, interior2]))
        
        # Config 4: Regular 10-gon + 3 interior points
        hull_pts = _regular_polygon(10, radius=1.0)
        interior = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([hull_pts, interior]))
        
        # Config 5: Regular 9-gon + 4 interior points
        hull_pts = _regular_polygon(9, radius=1.0)
        interior = _regular_polygon(4, radius=0.35)
        configs.append(np.vstack([hull_pts, interior]))
        
        # Config 6: Two concentric rings (7 outer + 6 inner)
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=0.5)
        configs.append(np.vstack([outer, inner]))
        
        # Config 7: 8 outer + 5 inner
        outer = _regular_polygon(8, radius=1.0)
        inner = _regular_polygon(5, radius=0.45)
        configs.append(np.vstack([outer, inner]))
        
        # Config 8: 9 outer + 4 inner
        outer = _regular_polygon(9, radius=1.0)
        inner = _regular_polygon(4, radius=0.4)
        configs.append(np.vstack([outer, inner]))
        
        # Config 9: 10 outer + 3 inner
        outer = _regular_polygon(10, radius=1.0)
        inner = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([outer, inner]))
        
        # Config 10: 11 outer + 2 inner
        outer = _regular_polygon(11, radius=1.0)
        inner = _regular_polygon(2, radius=0.25)
        configs.append(np.vstack([outer, inner]))
        
        return configs

    best_points = None
    best_score = -float('inf')
    
    configs = _make_initial_configurations()
    
    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 10:
            break
        
        # Evaluate initial config
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Try to optimize with L-BFGS-B
        if time.time() > start_time + time_limit - 10:
            break
        
        try:
            result = opt.minimize(
                objective_flat,
                config.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            optimized = result.x.reshape(-1, 2)
            opt_score, _ = compute_score(optimized)
            if opt_score > best_score:
                best_score = opt_score
                best_points = optimized.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead refinement on the best result so far
        if time.time() > start_time + time_limit - 10:
            break
        
        if best_points is not None:
            try:
                # Use Nelder-Mead for local refinement
                res = opt.minimize(
                    objective_flat,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score, _ = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass
    
    # If we have time, do a few more random restarts
    if time.time() < start_time + time_limit - 10:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > start_time + time_limit - 10:
                break
            # Random convex-ish configuration
            n_outer = np.random.randint(6, 12)
            n_inner = N - n_outer
            outer = _regular_polygon(n_outer, radius=np.random.uniform(0.8, 1.2))
            if n_inner > 0:
                inner = _regular_polygon(n_inner, radius=np.random.uniform(0.2, 0.5))
                config = np.vstack([outer, inner])
            else:
                config = outer
            config = config + np.random.normal(0, 0.1, config.shape)
            
            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()
            
            if time.time() > start_time + time_limit - 10:
                break
            
            try:
                result = opt.minimize(
                    objective_flat,
                    config.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
