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

def _fourier_radii_to_points(params, n=20):
    """
    Convert Fourier coefficients of radial deviations to points.
    params: [a1, b1, a2, b2, a3, b3]
    r_i = 1 + sum (a_k cos(2*pi*k*i/n) + b_k sin(2*pi*k*i/n))
    Angles are fixed at 2*pi*i/n.
    """
    angles = 2 * np.pi * np.arange(n) / n
    radii = np.ones(n)
    for k in range(1, len(params)//2 + 1):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        theta = 2 * np.pi * k * np.arange(n) / n
        radii += a * np.cos(theta) + b * np.sin(theta)
    
    # Ensure radii are positive
    if np.any(radii <= 0.1):
        return None
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _neg_score_from_radii(params):
    """Negative score for optimization."""
    pts = _fourier_radii_to_points(params)
    if pts is None:
        return 10.0 # Penalty for invalid radii
    
    # Check convexity quickly
    n = len(pts)
    centroid = np.mean(pts, axis=0)
    p = pts - centroid
    edges = np.roll(p, -1, axis=0) - p
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)):
        return 10.0 # Penalty for non-convex
    
    score = _compute_score(pts)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: regular polygon (all Fourier coefficients zero)
    params0 = np.zeros(6)
    best_points = _fourier_radii_to_points(params0)
    best_score = _compute_score(best_points)
    
    np.random.seed(42)
    
    # Phase 1: Differential evolution on 6 radial Fourier parameters
    # Bounds: keep radii in [0.5, 1.5] roughly
    bounds = [(-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05)]
    
    try:
        if time.time() < deadline - 120:
            result_de = differential_evolution(
                _neg_score_from_radii, bounds,
                seed=42, maxiter=300, tol=1e-8,
                polish=True, init='sobol',
                workers=1
            )
            if result_de.fun < -best_score:
                params_opt = result_de.x
                pts = _fourier_radii_to_points(params_opt)
                if pts is not None:
                    s = _compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
    except Exception:
        pass
    
    # Phase 2: Local refinement
    if time.time() < deadline - 60:
        init_candidates = [params0]
        if 'result_de' in locals() and result_de.fun < -best_score:
            init_candidates.append(result_de.x)
            
        # Add some random perturbations
        for trial in range(5):
            if time.time() > deadline - 40:
                break
            random_params = np.random.uniform(-0.1, 0.1, size=6)
            init_candidates.append(random_params)
        
        for init in init_candidates:
            if time.time() > deadline - 30:
                break
            try:
                result_local = minimize(
                    _neg_score_from_radii, init,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                params_local = result_local.x
                pts = _fourier_radii_to_points(params_local)
                if pts is not None:
                    s = _compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass
    
    # Phase 3: Final polish
    if time.time() < deadline - 10:
        try:
            if 'result_local' in locals() and result_local.fun < -best_score:
                init_final = result_local.x
            elif 'result_de' in locals() and result_de.fun < -best_score:
                init_final = result_de.x
            else:
                init_final = params0
            
            result_polish = minimize(
                _neg_score_from_radii, init_final,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts = _fourier_radii_to_points(result_polish.x)
            if pts is not None:
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
            
    return best_points
