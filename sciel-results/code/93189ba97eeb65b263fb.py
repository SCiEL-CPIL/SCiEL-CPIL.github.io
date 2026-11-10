# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices using radial Fourier perturbation,
    optimized via Differential Evolution followed by a local polish.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    bound = 0.5
    bounds = [(-bound, bound)] * 4
    
    def compute_polygon(params):
        a1, b1, a2, b2 = params
        r = 1.0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
        if np.any(r <= 0.01):
            return None
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)

    def is_valid_polygon(polygon):
        if polygon is None:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(polygon[i] - polygon[j]) < 1e-6:
                    return False
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return False
        edges = np.roll(centered, -1, axis=0) - centered
        cross_products = np.array([
            edges[i][0] * edges[(i+1)%n][1] - edges[i][1] * edges[(i+1)%n][0]
            for i in range(n)
        ])
        if np.any(np.abs(cross_products) <= 1e-9):
            return False
        if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return False
        return True

    def vertex_score(polygon, v_idx):
        v = polygon[v_idx]
        dists = np.linalg.norm(polygon - v, axis=1)
        other_dists = np.delete(dists, v_idx)
        sorted_dists = np.sort(other_dists)
        if sorted_dists[0] < 0.05:
            return 0.0
        gaps = np.diff(sorted_dists)
        k = int(np.argmin(gaps))
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        if k == 0:
            selected = sorted_dists[0:4]
        elif k == 17:
            selected = sorted_dists[15:19]
        else:
            selected = sorted_dists[k-1:k+3]
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        if raw < 1e-9:
            return 0.0
        return min(1.0, 1.0 / raw)

    def min_vertex_score(polygon):
        if polygon is None or not is_valid_polygon(polygon):
            return 0.0
        scores = [vertex_score(polygon, i) for i in range(n)]
        return np.min(scores)

    def objective(params):
        polygon = compute_polygon(params)
        if polygon is None:
            return 1000.0
        if not is_valid_polygon(polygon):
            return 1000.0
        scores = np.array([vertex_score(polygon, i) for i in range(n)])
        return -np.min(scores)

    start_time = time.time()
    timeout = 600.0
    margin = 30.0

    def time_check(params, convergence):
        if time.time() - start_time > timeout - margin:
            raise Exception("Timeout")

    best_params = np.zeros(4)
    best_score = 0.0

    # Phase 1: Differential Evolution
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=200, popsize=20,
            tol=1e-10, seed=42,
            callback=time_check
        )
        best_params = result.x
        best_score = -result.fun if result.fun < 500 else 0.0
    except Exception:
        pass

    # Phase 2: Local polish with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < timeout - margin - 30:
        try:
            def polish_obj(params):
                if time.time() - start_time > timeout - margin:
                    raise Exception("Timeout")
                return objective(params)
            
            polish_result = minimize(
                polish_obj, best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if polish_result.fun < objective(best_params):
                best_params = polish_result.x
                best_score = -polish_result.fun if polish_result.fun < 500 else 0.0
        except Exception:
            pass

    best_polygon = compute_polygon(best_params)
    if best_polygon is None or not is_valid_polygon(best_polygon):
        r = np.ones(n)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
    else:
        polygon = best_polygon

    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        r = np.ones(n)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)

    return centered / mean_dist

# EVOLVE-BLOCK-END
