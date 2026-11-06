import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 148.0
_SAFETY = 5.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _score(pts):
    """
    Calculate the score for a set of points.
    Score = Delta / n^n, where Delta is the product of distances of all ordered pairs
    after scaling the set so that the diameter is 2.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    dia = d.max()
    if dia < 1e-12:
        return 0.0
    
    # Set diagonal to 1 to avoid log(0)
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    log_prod_scaled = np.log(d_safe).sum() + _N * (_N - 1) * (math.log(2.0) - math.log(dia))
    log_score = log_prod_scaled - _N * math.log(_N)
    
    return math.exp(log_score)

def _log_score(pts):
    """Return log of score for optimization stability."""
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    dia = d.max()
    if dia < 1e-12:
        return -1e10
    
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    log_prod_scaled = np.log(d_safe).sum() + _N * (_N - 1) * (math.log(2.0) - math.log(dia))
    log_score = log_prod_scaled - _N * math.log(_N)
    
    return log_score

def _get_points_from_gaps(gaps):
    """
    Given 8 angular gaps (summing to 2*pi), generate 8 points on the unit circle.
    gaps[0] is the gap after point 0, etc.
    """
    gaps = np.asarray(gaps, dtype=float)
    # Normalize to sum to 2*pi
    total = gaps.sum()
    if total < 1e-12:
        return np.eye(_N, 2)  # fallback
    gaps = gaps * (2.0 * math.pi / total)
    
    angles = np.zeros(_N)
    angles[0] = 0.0
    for i in range(1, _N):
        angles[i] = angles[i-1] + gaps[i-1]
    
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _objective_circular(gaps):
    """
    Objective for circular optimization: minimize -log_score.
    gaps: 7 values (the 8th is determined by 2*pi - sum of first 7)
    """
    if _remaining() < 2:
        return 1e10
    
    # Reconstruct 8 gaps
    all_gaps = np.zeros(_N)
    all_gaps[:7] = gaps
    all_gaps[7] = 2.0 * math.pi - np.sum(gaps)
    
    # All gaps must be positive
    if np.any(all_gaps < 1e-8):
        return 1e10
    
    pts = _get_points_from_gaps(all_gaps)
    
    # Check for distinct points
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e10
    
    ls = _log_score(pts)
    return -ls

def _refine_generic(pts, maxiter=2000, ftol=1e-12):
    """
    Refine a full 8-point configuration using L-BFGS-B.
    """
    def obj(x):
        p = x.reshape(_N, 2)
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return 1e12
        return -_log_score(p)
        
    try:
        res = minimize(obj, pts.ravel(), method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
        cand = res.x.reshape(_N, 2)
        d = np.linalg.norm(cand[:, np.newaxis, :] - cand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return pts, _score(pts)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)

def _refine_penalty(pts, maxiter=1000, t=50.0):
    """
    Refine using penalty method for diameter constraint.
    """
    def neg_log_score_constrained(x):
        pts = x.reshape(_N, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(_N, k=1)
        d_ij = dists[iu]
        
        log_s = _log_score(pts)
        
        # Soft penalty for distances exceeding 2
        penalty = np.sum(np.logaddexp(0, (d_ij - 2.0) * 100.0) / 100.0)
        
        return -log_s + t * penalty
    
    try:
        res = minimize(neg_log_score_constrained, pts.ravel(), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        cand = res.x.reshape(_N, 2)
        d = np.linalg.norm(cand[:, np.newaxis, :] - cand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return pts, _score(pts)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)

def _get_points_D2(params):
    """
    Generate 8 points with D2 symmetry.
    params: [a, b, c, d]
    Points: (±a, ±b) and (±c, ±d)
    """
    a, b, c, d = params
    p1 = np.array([a, b])
    p2 = np.array([-a, b])
    p3 = np.array([-a, -b])
    p4 = np.array([a, -b])
    p5 = np.array([c, d])
    p6 = np.array([-c, d])
    p7 = np.array([-c, -d])
    p8 = np.array([c, -d])
    return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

def _objective_D2(params):
    pts = _get_points_D2(params)
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
    return -_log_score(pts)

def construct():
    t0 = time.time()
    best_pts = None
    best_score = 0.0
    best_log_score = -1e10
    
    rng = np.random.default_rng(12345)
    
    # ============================================================
    # PHASE 1: Circular gap optimization (7D)
    # ============================================================
    
    # Start with regular octagon gaps
    regular_gaps = np.full(7, 2.0 * math.pi / _N)
    
    # Try multiple gap configurations
    gap_starts = [regular_gaps.copy()]
    
    # Add perturbed regular gaps
    for _ in range(10):
        perturbed = regular_gaps + rng.normal(0, 0.1, size=7)
        perturbed = np.abs(perturbed) + 0.05
        gap_starts.append(perturbed)
    
    # Add some structured non-regular starts
    # Two close pairs, two far pairs
    structured = np.array([0.3, 0.3, 0.8, 0.8, 0.5, 0.5, 0.8])
    structured = structured / structured.sum() * 2.0 * math.pi
    gap_starts.append(structured[:7])
    
    # Alternating close/far
    alt = np.array([0.2, 1.0, 0.2, 1.0, 0.2, 1.0, 0.2])
    alt = alt / alt.sum() * 2.0 * math.pi
    gap_starts.append(alt)
    
    for gap_start in gap_starts:
        if _remaining() < 15:
            break
        
        try:
            res = minimize(_objective_circular, gap_start, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < 1e9:
                all_gaps = np.zeros(_N)
                all_gaps[:7] = res.x
                all_gaps[7] = 2.0 * math.pi - np.sum(res.x)
                
                if np.all(all_gaps > 1e-8):
                    pts = _get_points_from_gaps(all_gaps)
                    s = _score(pts)
                    ls = _log_score(pts)
                    if s > best_score:
                        best_score = s
                        best_log_score = ls
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # ============================================================
    # PHASE 2: D2 symmetric optimization (4D)
    # ============================================================
    
    if _remaining() > 20:
        bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
        
        try:
            res_de = differential_evolution(
                _objective_D2, bounds, 
                seed=42, 
                maxiter=80, 
                tol=1e-8,
                polish=True,
                popsize=12
            )
            if res_de.fun < 1e10:
                pts = _get_points_D2(res_de.x)
                s = _score(pts)
                ls = _log_score(pts)
                if s > best_score:
                    best_score = s
                    best_log_score = ls
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # ============================================================
    # PHASE 3: Generic refinement from best found so far
    # ============================================================
    
    if best_pts is not None and _remaining() > 25:
        # Refine with L-BFGS-B
        cand, s = _refine_generic(best_pts, maxiter=1000)
        if s > best_score:
            best_score = s
            best_pts = cand
        
        # Refine with penalty method
        if _remaining() > 20:
            cand, s = _refine_penalty(best_pts, maxiter=500, t=50.0)
            if s > best_score:
                best_score = s
                best_pts = cand
    
    # ============================================================
    # PHASE 4: Multiple random D2 starts with refinement
    # ============================================================
    
    num_random_starts = 15
    for i in range(num_random_starts):
        if _remaining() < 20:
            break
        
        a = rng.uniform(0.5, 5.0)
        b = rng.uniform(0.5, 5.0)
        c = rng.uniform(0.5, 5.0)
        d = rng.uniform(0.5, 5.0)
        
        params = [a, b, c, d]
        pts = _get_points_D2(params)
        
        cand, s = _refine_generic(pts, maxiter=200)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # ============================================================
    # PHASE 5: Noise-based refinement from best
    # ============================================================
    
    if best_pts is not None and _remaining() > 40:
        for i in range(8):
            if _remaining() < 15:
                break
            
            noisy = best_pts + rng.normal(0, 0.3, size=(_N, 2))
            cand, s = _refine_generic(noisy, maxiter=500)
            if s > best_score:
                best_score = s
                best_pts = cand
            
            # Also try penalty refinement
            if _remaining() > 15:
                cand, s = _refine_penalty(noisy, maxiter=300, t=100.0)
                if s > best_score:
                    best_score = s
                    best_pts = cand
    
    # ============================================================
    # PHASE 6: Final polish
    # ============================================================
    
    if best_pts is not None and _remaining() > 10:
        cand, s = _refine_generic(best_pts, maxiter=2000, ftol=1e-14)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # ============================================================
    # Fallback to regular octagon if nothing worked
    # ============================================================
    
    if best_pts is None:
        best_pts = np.array([[math.cos(2*math.pi*k/_N), math.sin(2*math.pi*k/_N)] for k in range(_N)])
        best_score = _score(best_pts)
    
    # Final check and cleanup
    best_pts = np.asarray(best_pts, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-5:
        for i in range(_N):
            for j in range(i+1, _N):
                if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-5:
                    best_pts[j] += [1e-5, 0]
    
    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e6
    
    return [[float(x), float(y)] for (x, y) in best_pts]
