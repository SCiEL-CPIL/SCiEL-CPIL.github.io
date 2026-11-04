"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return float(np.min(areas)) / hull_area

def _soft_min_objective(x, beta=100.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    # Add penalty if hull area is too small to encourage larger hulls
    hull_penalty = 0.0
    if hull_area < 0.3:
        hull_penalty = (0.3 - hull_area) * 200
    
    return -soft_min_area / hull_area + hull_penalty

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _initial_points(seed=0):
    """Generate initial points: regular polygon with slight perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence, scaled to unit circle."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    pts = np.column_stack([x, y])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    pts = pts / radii[:, np.newaxis]
    return pts

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _three_ring(n1, n2, n3, r1=1.0, r2=0.6, r3=0.3):
    """Points on three concentric rings."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    return np.vstack([p1, p2, p3])

def _elliptic_distortion(pts, matrix):
    """Apply elliptical distortion defined by symmetric 2x2 matrix to points."""
    # matrix = [[a, b], [b, c]]
    # Apply: pts @ matrix.T
    return pts @ matrix.T

def _elliptic_objective(params, base_pts, beta=200.0):
    """
    Objective for optimizing elliptical distortion parameters.
    params = [a, b, c] for matrix [[a, b], [b, c]]
    """
    a, b, c = params
    matrix = np.array([[a, b], [b, c]])
    
    # Ensure positive definite (a > 0, ac > b^2)
    if a <= 0 or a * c <= b * b:
        return 1e6
    
    distorted_pts = _elliptic_distortion(base_pts, matrix)
    return _soft_min_objective(distorted_pts.flatten(), beta)

def _nelder_mead_polish(pts, max_iter=2000, tol=1e-12):
    """Nelder-Mead polish on the exact objective."""
    try:
        res = minimize(
            _exact_objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts.copy()

def _lbfgs_polish(pts, max_iter=3000, beta=500.0):
    """L-BFGS-B polish with high beta for sharpness."""
    try:
        res = minimize(
            _soft_min_objective,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts.copy()

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _initial_points(0)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_initial_points(0))
    
    # 2. Concentric rings (various splits and radii)
    for n_outer in range(6, 11):
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Three-ring configurations
    for n1 in range(5, 9):
        for n2 in range(3, 6):
            n3 = N - n1 - n2
            if n3 >= 2:
                for r2 in [0.35, 0.45, 0.55, 0.65]:
                    for r3 in [0.15, 0.25, 0.35, 0.45]:
                        candidates.append(_three_ring(n1, n2, n3, 1.0, r2, r3))
    
    # 4. Halton sequence points
    candidates.append(_halton_points(N))
    
    # 5. Random points on/near circle
    rng = np.random.default_rng(42)
    for i in range(40):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.4, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon
    for i in range(40):
        noise = rng.normal(0, 0.15, (N, 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(base + noise)
        
    # 7. "Grid-like" or structured perturbations
    for i in range(20):
        pts = np.zeros((N, 2))
        for j in range(N):
            angle = 2 * np.pi * j / N + rng.normal(0, 0.15)
            radius = 0.7 + rng.normal(0, 0.15)
            pts[j, 0] = radius * np.cos(angle)
            pts[j, 1] = radius * np.sin(angle)
        candidates.append(pts)
    
    # 8. Elliptic distortions of regular polygon
    base_reg = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)),
                                  np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))])
    for a in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        for c in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            for b in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                matrix = np.array([[a, b], [b, c]])
                if a > 0 and a * c > b * b:
                    distorted = _elliptic_distortion(base_reg, matrix)
                    candidates.append(distorted)
    
    # Evaluate all candidates with exact score
    scored_candidates = []
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        scored_candidates.append((s, pts))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Sort candidates by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top candidates for refinement
    top_candidates = [pts for _, pts in scored_candidates[:min(25, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [30.0, 80.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        # Number of candidates to refine decreases as we get more precise
        num_to_refine = min(10, len(top_candidates)) if beta < 100 else min(5, len(top_candidates))
        
        for init_pts in top_candidates[:num_to_refine]:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 150}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    
            except Exception:
                pass
        
        # Update top candidates: keep the best one and a few others
        if best_pts is not None:
            top_candidates = [best_pts.copy()] + [p for p in top_candidates if not np.allclose(p, best_pts, atol=1e-6)][:8]
    
    # Elliptic distortion refinement on best point
    if time.time() < deadline - 20:
        try:
            # Optimize elliptic distortion of the best points
            base_for_elliptic = best_pts.copy()
            # Center the points
            centroid = np.mean(base_for_elliptic, axis=0)
            base_for_elliptic = base_for_elliptic - centroid
            
            result_elliptic = minimize(
                _elliptic_objective,
                np.array([1.0, 0.0, 1.0]),
                args=(base_for_elliptic, 300.0),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            a, b, c = result_elliptic.x
            matrix = np.array([[a, b], [b, c]])
            if a > 0 and a * c > b * b:
                distorted = _elliptic_distortion(base_for_elliptic, matrix) + centroid
                distorted_score = _score(distorted)
                if distorted_score > best_score:
                    best_score = distorted_score
                    best_pts = distorted.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish on the best point
    if time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=8000, tol=1e-14)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished
    
    # L-BFGS-B polish with high beta
    if time.time() < deadline - 10:
        polished = _lbfgs_polish(best_pts, max_iter=5000, beta=1000.0)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished
    
    # One more Nelder-Mead pass if time allows
    if time.time() < deadline - 5:
        final_polished = _nelder_mead_polish(best_pts, max_iter=5000, tol=1e-15)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_polished
    
    # Final L-BFGS pass
    if time.time() < deadline - 2:
        final_lbfgs = _lbfgs_polish(best_pts, max_iter=3000, beta=2000.0)
        final_lbfgs_score = _score(final_lbfgs)
        if final_lbfgs_score > best_score:
            best_score = final_lbfgs_score
            best_pts = final_lbfgs
    
    return best_pts
# EVOLVE-BLOCK-END
