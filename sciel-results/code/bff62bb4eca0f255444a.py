# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _fourier_to_pts(coeffs, n=20):
    """
    Convert Fourier coefficients to points.
    coeffs = [a0, a2, b2, a4, b4, ...]
    We use only even harmonics to maintain symmetry.
    r(theta) = a0 + sum_{k=1}^M (a_{2k} cos(2k theta) + b_{2k} sin(2k theta))
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.full(n, coeffs[0])
    # coeffs[0] is a0
    # coeffs[1:] are pairs (a, b) for harmonics 2, 4, 6...
    num_harmonics = len(coeffs) - 1
    for i in range(num_harmonics):
        k = i + 1
        harmonic = 2 * k
        a = coeffs[1 + 2 * i]
        b = coeffs[2 + 2 * i]
        radii += a * np.cos(harmonic * angles) + b * np.sin(harmonic * angles)
    
    # Ensure radii are positive to maintain convexity/validity
    radii = np.maximum(radii, 0.5)
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _smooth_score_fourier(coeffs):
    """
    Smoothed surrogate for the min-vertex-score.
    """
    n = 20
    pts = _fourier_to_pts(coeffs, n)
    
    pn, _, _ = _normalize(pts)
    if pn is None:
        return -100.0
        
    if not _check_strictly_convex(pn):
        return -100.0

    # Calculate vertex scores
    scores = []
    for i in range(n):
        vs = _vertex_score(pn, i)
        scores.append(vs)
    
    scores = np.array(scores)
    
    # Soft-min
    tau = 0.05
    neg_scores_over_tau = -scores / tau
    max_neg = np.max(neg_scores_over_tau)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores_over_tau - max_neg)))
    soft_min = -tau * log_sum_exp
    
    return -soft_min

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    # Initial guess: Regular polygon
    # a0 = 1, others 0
    initial_coeffs = np.zeros(1 + 2 * 5) # 5 harmonics: 2,4,6,8,10
    initial_coeffs[0] = 1.0
    
    best_score = 0.0
    best_coeffs = initial_coeffs.copy()
    
    def try_candidate(coeffs):
        nonlocal best_score, best_coeffs
        pts = _fourier_to_pts(coeffs, n)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_coeffs = coeffs.copy()

    # Try a few different initializations of the Fourier coefficients
    # We want to explore shapes that might have more "equidistant" properties
    seeds = [initial_coeffs]
    
    # Add some small perturbations
    for amp in [0.05, 0.1]:
        for k in range(1, 4):
            c = initial_coeffs.copy()
            # Perturb harmonic 2k
            idx_a = 1 + 2 * (k - 1)
            idx_b = 2 + 2 * (k - 1)
            c[idx_a] = amp
            seeds.append(c)
            c[idx_b] = amp
            seeds.append(c)
            
    # Run Nelder-Mead from multiple seeds
    max_iter_per_seed = 3000
    tol = 1e-8
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        try_candidate(seed)
        
        try:
            res = minimize(
                _smooth_score_fourier, 
                seed, 
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_seed, 'xatol': tol, 'fatol': tol}
            )
            
            # Evaluate the result
            try_candidate(res.x)
            
        except Exception as e:
            pass
            
    # Final polish: Local search on the best coefficients
    if best_score > 0 and time.time() < deadline:
        current = best_coeffs.copy()
        step = 0.01
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                for delta in [step, -step]:
                    new_c = current.copy()
                    new_c[i] += delta
                    s_new = _score_polygon(_fourier_to_pts(new_c, n))
                    if s_new > best_score:
                        best_score = s_new
                        best_coeffs = new_c.copy()
                        current = new_c
                        improved = True
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break

    return _fourier_to_pts(best_coeffs, n)

# EVOLVE-BLOCK-END
