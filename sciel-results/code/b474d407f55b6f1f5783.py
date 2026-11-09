import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

def _compute_score(points):
    """Compute the minimum vertex score for a given polygon."""
    n = len(points)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Check strict convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d_sorted = np.sort(np.delete(d, i))
        if np.min(d_sorted) < 0.05:
            min_score = 0.0
            break
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
    return min_score

def _fourier_to_points(params, n=20):
    """Convert Fourier mode coefficients to polygon vertices.
    params: [a1, b1, a2, b2, a3, b3, a4, b4]
    Returns (n, 2) array of points.
    """
    base_angle = 2 * np.pi / n
    angles = np.full(n, base_angle)
    for k in range(1, 5):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        angles += a * np.cos(theta) + b * np.sin(theta)
    
    # Ensure angles are positive
    angles = np.maximum(angles, 0.01)
    
    # Normalize to sum to 2*pi
    angles = angles / np.sum(angles) * 2 * np.pi
    
    # Cumulative angles
    cum_angles = np.cumsum(angles)
    cum_angles = np.append([0], cum_angles)
    
    x = np.cos(cum_angles[:-1])
    y = np.sin(cum_angles[:-1])
    return np.stack([x, y], axis=1)

def _score_from_fourier(params):
    pts = _fourier_to_points(params)
    return _compute_score(pts)

def _neg_score_from_fourier(params):
    return -_score_from_fourier(params)

def _soft_neg_score_from_fourier(params, temperature=0.05):
    """Soft version of the score for smoother optimization."""
    pts = _fourier_to_points(params)
    n = len(pts)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 1.0
    pts = pts / mean_dist
    
    # Check convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 1.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 1.0
    
    # Compute soft min of vertex scores
    vertex_scores = []
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d_sorted = np.sort(np.delete(d, i))
        if np.min(d_sorted) < 0.05:
            vertex_scores.append(0.0)
            continue
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            vertex_scores.append(0.0)
            continue
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        vertex_scores.append(score)
    
    # Soft min using log-sum-exp
    vertex_scores = np.array(vertex_scores)
    log_sum = np.log(np.sum(np.exp(-vertex_scores / temperature)))
    soft_min = -temperature * log_sum
    return -soft_min

def _radial_to_points(radii, n=20):
    """Convert radial distances to polygon vertices using equal angular spacing."""
    base_angle = 2 * np.pi / n
    angles = np.arange(n) * base_angle
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _score_from_radial(radii):
    pts = _radial_to_points(radii)
    return _compute_score(pts)

def _neg_score_from_radial(radii):
    return -_score_from_radial(radii)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: regular polygon
    angles0 = np.ones(n)
    best_points = _fourier_to_points(np.zeros(8))
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Differential evolution on Fourier parameters
    try:
        if time.time() < deadline - 120:
            bounds = [(-0.3, 0.3)] * 8
            result = differential_evolution(
                _neg_score_from_fourier, bounds,
                seed=42, maxiter=200, tol=1e-8,
                polish=True, init='sobol',
                workers=1, popsize=30
            )
            if result.fun < -best_score:
                params_opt = result.x
                pts = _fourier_to_points(params_opt)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    except Exception:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            # Get current best params
            if 'result' in locals() and result.fun < -best_score:
                init_params = result.x
            else:
                init_params = np.zeros(8)
            
            result_local = minimize(_neg_score_from_fourier, init_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            pts = _fourier_to_points(result_local.x)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
    
    # Phase 3: Multi-start with different seeds
    if time.time() < deadline - 30:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 30:
                break
            try:
                np.random.seed(seed)
                bounds = [(-0.3, 0.3)] * 8
                result = differential_evolution(
                    _neg_score_from_fourier, bounds,
                    seed=seed, maxiter=50, tol=1e-8,
                    polish=False, init='sobol',
                    workers=1, popsize=20
                )
                if result.fun < -best_score:
                    params_opt = result.x
                    pts = _fourier_to_points(params_opt)
                    s = _compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass
    
    return best_points
