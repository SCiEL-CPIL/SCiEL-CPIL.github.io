# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _vertex_score(distances):
    """Compute the vertex score from sorted distances to other 19 vertices."""
    d = np.sort(distances)
    n = len(d)
    # Check for any distance < 0.05
    if d[0] < 0.05:
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select 4 distances
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k - 1, k, k + 1, k + 2]
    four_d = d[idx]
    raw = np.mean(np.maximum(four_d / D, D / four_d))
    return min(1.0, 1.0 / raw) if raw > 0 else 0.0


def _compute_scores(polygon):
    """Compute per-vertex scores for a (20, 2) polygon."""
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        diffs = polygon - polygon[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.delete(dists, i)  # exclude self
        scores[i] = _vertex_score(dists)
    return scores


def _soft_min(scores, temperature=0.01):
    """Soft minimum using log-sum-exp trick."""
    scores = np.asarray(scores)
    return -temperature * np.log(np.sum(np.exp(-scores / temperature)))


def _evaluate(polygon):
    """Evaluate the polygon: check validity and return min vertex score."""
    n = len(polygon)
    # Check finite
    if not np.all(np.isfinite(polygon)):
        return 0.0
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return 0.0
    # Check strict convexity: all turns same sign and nonzero
    cross_products = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0
    # Check total turning ~ 2*pi (for convex, should be exactly 2*pi in magnitude)
    # For a convex polygon, the sum of exterior angles is 2*pi
    # We can check that the polygon is simple and convex via cross products sign
    # Already checked above.
    
    scores = _compute_scores(polygon)
    return np.min(scores)


def _radial_polygon(params, n=20):
    """Generate a convex polygon from radial harmonic parameters.
    params: [a1, phi1, a2, phi2, ..., aK, phiK]
    r(theta) = 1 + sum_k a_k * cos(k*theta + phi_k)
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    K = len(params) // 2
    for k in range(K):
        a = params[2 * k]
        phi = params[2 * k + 1]
        r += a * np.cos((k + 1) * angles + phi)
    # Ensure convexity: r must be positive and the polygon must be convex
    # For small a, this is guaranteed. We'll clip r to be positive.
    r = np.maximum(r, 0.1)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.stack([x, y], axis=1)


def _soft_objective(params, n=20, temperature=0.01):
    """Soft-min objective for optimization (to be minimized, so negate)."""
    polygon = _radial_polygon(params, n)
    if not np.all(np.isfinite(polygon)):
        return 1e10
    scores = _compute_scores(polygon)
    soft_min_val = _soft_min(scores, temperature)
    return -soft_min_val  # negate because we maximize


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, leave margin
    
    # Start from regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    best_polygon = regular.copy()
    best_score = _evaluate(regular)
    
    # Try optimization with radial harmonics
    # Use a few harmonics for flexibility
    K = 5  # number of harmonics
    n_params = 2 * K
    # Initial params: all zeros (regular polygon)
    x0 = np.zeros(n_params)
    
    # Bounds: keep amplitudes small to ensure convexity
    bounds = [(-0.1, 0.1), (-np.pi, np.pi)] * K
    
    # Optimization loop with time check
    max_iter = 500
    for attempt in range(3):  # up to 3 restarts
        if time.time() > deadline - 60.0:
            break
        
        # Perturb initial guess
        if attempt == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + 0.05 * np.random.randn(n_params)
            x_init[1::2] = np.clip(x_init[1::2], -np.pi, np.pi)
            x_init[0::2] = np.clip(x_init[0::2], -0.1, 0.1)
        
        # Use L-BFGS-B
        try:
            result = minimize(
                _soft_objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        polygon = _radial_polygon(result.x, n)
        score = _evaluate(polygon)
        
        if score > best_score:
            best_score = score
            best_polygon = polygon.copy()
        
        if time.time() > deadline - 30.0:
            break
    
    # Final polish: try small random perturbations around best
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            perturbed = best_polygon + 0.001 * np.random.randn(n, 2)
            # Re-center and normalize
            centroid = np.mean(perturbed, axis=0)
            perturbed = perturbed - centroid
            dists = np.linalg.norm(perturbed, axis=1)
            mean_dist = np.mean(dists)
            if mean_dist > 0.01:
                perturbed = perturbed / mean_dist
            score = _evaluate(perturbed)
            if score > best_score:
                best_score = score
                best_polygon = perturbed.copy()
    
    return best_polygon

# EVOLVE-BLOCK-END
