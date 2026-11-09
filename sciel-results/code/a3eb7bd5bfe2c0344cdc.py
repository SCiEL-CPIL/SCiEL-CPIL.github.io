import numpy as np
import time
from scipy.optimize import minimize

def _angles_to_points(thetas):
    """Convert angles to 2D points on unit circle."""
    thetas = np.asarray(thetas, dtype=np.float64)
    return np.column_stack((np.cos(thetas), np.sin(thetas)))

def _normalize(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(points):
    """Check if points form a strictly convex polygon in cyclic order."""
    n = len(points)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return False
    # Check all turns are nonzero and same sign
    crosses = np.empty(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses[i] = v1[0]*v2[1] - v1[1]*v2[0]
    if np.any(np.abs(crosses) < 1e-14):
        return False
    if len(np.unique(np.sign(crosses))) != 1:
        return False
    return True

def _vertex_score(v_idx, points):
    """Compute the score for a single vertex."""
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
    
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_score(points):
    """Compute the minimum vertex score over all vertices."""
    n = len(points)
    scores = np.array([_vertex_score(i, points) for i in range(n)])
    return np.min(scores)

def _soft_min(scores, beta):
    """Smooth approximation of min using log-sum-exp."""
    if len(scores) == 0:
        return 0.0
    shifted = -beta * scores
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return 0.0
    result = -max_shifted/beta - np.log(np.sum(np.exp(shifted - max_shifted)))/beta
    return result

def _neg_soft_score_from_angles(thetas, beta):
    """Negative soft-min of vertex scores, given angles."""
    points = _angles_to_points(thetas)
    normalized = _normalize(points)
    if normalized is None:
        return 0.0
    if not _is_strictly_convex(normalized):
        return 0.0
    n = len(normalized)
    scores = np.array([_vertex_score(i, normalized) for i in range(n)])
    if np.any(scores == 0):
        return 0.0
    return -_soft_min(scores, beta)

def _neg_exact_score_from_angles(thetas):
    """Negative exact min vertex score, given angles."""
    points = _angles_to_points(thetas)
    normalized = _normalize(points)
    if normalized is None:
        return 0.0
    if not _is_strictly_convex(normalized):
        return 0.0
    s = _polygon_score(normalized)
    return -s

def _regular_angles(n=20):
    return np.linspace(0, 2*np.pi, n, endpoint=False)

def _jittered_angles(n=20, scale=0.01, seed=42):
    rng = np.random.default_rng(seed)
    base = _regular_angles(n)
    jitter = rng.normal(0, scale, size=n)
    thetas = base + jitter
    thetas = np.sort(thetas % (2*np.pi))
    return thetas

def _fourier_jittered_angles(n=20, seed=42):
    """Generate angles with Fourier series perturbation for non-uniform spacing."""
    rng = np.random.default_rng(seed)
    base = _regular_angles(n)
    # Use a few Fourier modes to create structured non-uniform spacing
    n_modes = 3
    amplitudes = rng.uniform(0.01, 0.05, size=n_modes)
    phases = rng.uniform(0, 2*np.pi, size=n_modes)
    t = np.arange(n)
    jitter = np.zeros(n)
    for m in range(1, n_modes+1):
        jitter += amplitudes[m-1] * np.sin(m * 2*np.pi * t / n + phases[m-1])
    thetas = base + jitter
    thetas = np.sort(thetas % (2*np.pi))
    return thetas

def equidistant_points_polygon20() -> np.ndarray:
    """Main function: optimize the 20-gon vertex positions."""
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit
    
    n = 20
    best_thetas = _regular_angles(n)
    best_score = _polygon_score(_normalize(_angles_to_points(best_thetas)))
    
    rng = np.random.default_rng(12345)
    
    def time_ok():
        return time.time() < deadline
    
    # Phase 1: Coarse search with smooth objective using Fourier-structured initializations
    beta_coarse = 15.0
    n_coarse = 15
    for restart in range(n_coarse):
        if not time_ok():
            break
        if restart == 0:
            x0 = _regular_angles(n)
        elif restart < 5:
            x0 = _fourier_jittered_angles(n, seed=rng.integers(0, 100000))
        else:
            x0 = _jittered_angles(n, scale=0.05, seed=rng.integers(0, 100000))
        
        try:
            res = minimize(
                lambda thetas: _neg_soft_score_from_angles(thetas, beta_coarse),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
            )
            if res.x is not None:
                s = -_neg_exact_score_from_angles(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Medium refinement
    beta_med = 40.0
    n_med = 10
    for restart in range(n_med):
        if not time_ok():
            break
        if restart == 0:
            x0 = best_thetas.copy()
        elif restart < 4:
            x0 = _fourier_jittered_angles(n, seed=rng.integers(0, 100000))
        else:
            x0 = best_thetas + rng.normal(0, 0.02, size=n)
            x0 = np.sort(x0 % (2*np.pi))
        
        try:
            res = minimize(
                lambda thetas: _neg_soft_score_from_angles(thetas, beta_med),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 60000}
            )
            if res.x is not None:
                s = -_neg_exact_score_from_angles(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine refinement with higher beta
    beta_fine = 80.0
    n_fine = 6
    for restart in range(n_fine):
        if not time_ok():
            break
        if restart == 0:
            x0 = best_thetas.copy()
        else:
            x0 = best_thetas + rng.normal(0, 0.005, size=n)
            x0 = np.sort(x0 % (2*np.pi))
        
        try:
            res = minimize(
                lambda thetas: _neg_soft_score_from_angles(thetas, beta_fine),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 60000}
            )
            if res.x is not None:
                s = -_neg_exact_score_from_angles(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Exact objective polish
    if time_ok():
        n_exact = 4
        for restart in range(n_exact):
            if not time_ok():
                break
            if restart == 0:
                x0 = best_thetas.copy()
            else:
                x0 = best_thetas + rng.normal(0, 0.002, size=n)
                x0 = np.sort(x0 % (2*np.pi))
            
            try:
                res = minimize(
                    _neg_exact_score_from_angles,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxfun': 80000}
                )
                if res.x is not None:
                    s = -_neg_exact_score_from_angles(res.x)
                    if s > best_score:
                        best_score = s
                        best_thetas = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: Nelder-Mead local search on exact objective
    if time_ok():
        x0 = best_thetas.copy()
        try:
            res = minimize(
                _neg_exact_score_from_angles,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.x is not None:
                s = -_neg_exact_score_from_angles(res.x)
                if s > best_score:
                    best_score = s
                    best_thetas = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Coordinate descent polish with adaptive steps
    if time_ok():
        current = best_thetas.copy()
        improved = True
        max_rounds = 50
        for rnd in range(max_rounds):
            if not time_ok() or not improved:
                break
            improved = False
            for i in range(n):
                if not time_ok():
                    break
                for delta in [0.001, -0.001, 0.0001, -0.0001, 0.00001, -0.00001]:
                    trial = current.copy()
                    trial[i] += delta
                    trial = np.sort(trial % (2*np.pi))
                    s = -_neg_exact_score_from_angles(trial)
                    if s > best_score:
                        best_score = -s
                        best_thetas = trial.copy()
                        current = trial.copy()
                        improved = True
                        break
    
    # Final: ensure normalized and convex
    final_points = _angles_to_points(best_thetas)
    normalized = _normalize(final_points)
    if normalized is not None and _is_strictly_convex(normalized):
        return normalized
    else:
        regular = _angles_to_points(_regular_angles(n))
        return _normalize(regular)
