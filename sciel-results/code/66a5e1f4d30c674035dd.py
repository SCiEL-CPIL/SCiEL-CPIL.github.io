# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex score.
    Uses radial Fourier perturbation of a regular polygon, optimized via
    multi-start Nelder-Mead local optimization.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, use 600s as safety margin

    # Fourier basis: low-frequency radial perturbations
    # r_i = 1 + sum_{k=1}^{K} (a_k cos(k*theta_i) + b_k sin(k*theta_i))
    K = 5  # number of Fourier harmonics (10 parameters total)
    n_params = 2 * K

    def theta_angles():
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def params_to_polygon(params):
        """Convert Fourier params to (n,2) polygon vertices."""
        angles = theta_angles()
        r = np.ones(n)
        for k in range(1, K + 1):
            a = params[2 * (k - 1)]
            b = params[2 * (k - 1) + 1]
            r += a * np.cos(k * angles) + b * np.sin(k * angles)
        # Ensure positive radii
        r = np.maximum(r, 0.1)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)

    def is_strictly_convex(polygon, tol=1e-10):
        """Check if polygon is strictly convex (CCW, all cross products positive)."""
        n_pts = len(polygon)
        for i in range(n_pts):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n_pts]
            p2 = polygon[(i + 2) % n_pts]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if cross <= tol:
                return False
        return True

    def vertex_score(v_idx, polygon):
        """Compute the score for a single vertex."""
        v = polygon[v_idx]
        # Distances to all other vertices
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        # Remove self-distance
        mask = np.ones(n, dtype=bool)
        mask[v_idx] = False
        dists = dists[mask]
        dists_sorted = np.sort(dists)

        # Check minimum distance
        if dists_sorted[0] < 0.05:
            return 0.0

        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists_sorted)
        k = int(np.argmin(diffs))
        D = (dists_sorted[k] + dists_sorted[k + 1]) / 2.0
        if D < 0.1:
            return 0.0

        # Select four distances
        if k == 0:
            selected = dists_sorted[0:4]
        elif k == 17:
            selected = dists_sorted[15:19]
        else:
            selected = dists_sorted[k - 1:k + 3]

        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        return min(1.0, 1.0 / raw)

    def min_vertex_score(polygon):
        """Minimum vertex score over all 20 vertices."""
        scores = [vertex_score(i, polygon) for i in range(n)]
        return min(scores)

    def objective(params):
        """Objective to minimize: negative of min vertex score, with convexity penalty."""
        polygon = params_to_polygon(params)
        if not is_strictly_convex(polygon):
            # Heavy penalty for non-convex
            return 1.0 + np.sum(params ** 2)
        score = min_vertex_score(polygon)
        return -score

    # Regular polygon as fallback
    angles = theta_angles()
    regular = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = min_vertex_score(regular)
    best_polygon = regular.copy()
    
    # Initial best params (zero perturbation)
    best_params = np.zeros(n_params)

    # Multi-start Nelder-Mead
    # We will try several random starts
    np.random.seed(42)
    n_starts = 20
    bounds_range = 0.15
    
    for i in range(n_starts):
        if time.time() > deadline - 60:
            break
            
        if i == 0:
            start_params = np.zeros(n_params)
        else:
            start_params = np.random.uniform(-bounds_range, bounds_range, n_params)
            
        try:
            # Nelder-Mead is good for low dimensions
            res = minimize(objective, start_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 0: # Found a valid convex polygon with positive score
                candidate_polygon = params_to_polygon(res.x)
                if is_strictly_convex(candidate_polygon):
                    candidate_score = min_vertex_score(candidate_polygon)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_polygon = candidate_polygon
                        best_params = res.x
        except Exception:
            pass

    return best_polygon

# EVOLVE-BLOCK-END
