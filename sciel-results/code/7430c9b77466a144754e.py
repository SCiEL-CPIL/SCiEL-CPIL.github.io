import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _normalize(pts):
    """Subtract centroid, divide by mean distance to centroid."""
    c = pts.mean(axis=0)
    p = pts - c
    d = np.linalg.norm(p, axis=1)
    m = d.mean()
    if m <= 0.01:
        return p, m
    return p / m, m

def _is_strictly_convex(pts):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turn ±2π."""
    n = len(pts)
    for i in range(n):
        if np.linalg.norm(pts[i] - pts[(i+1) % n]) < 1e-10:
            return False
    turns = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return False
        turns.append(cross)
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        return False
    # Total turning
    angles = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        da = a2 - a1
        while da > np.pi: da -= 2*np.pi
        while da < -np.pi: da += 2*np.pi
        angles.append(da)
    total = sum(angles)
    if abs(abs(total) - 2*np.pi) > 1e-6:
        return False
    return True

def _vertex_score(pts, i):
    """Compute score for vertex i."""
    n = len(pts)
    v = pts[i]
    dists = []
    for j in range(n):
        if j != i:
            dists.append(np.linalg.norm(v - pts[j]))
    dists = np.sort(dists)
    # 19 distances
    if any(d < 0.05 for d in dists):
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = dists[1:] - dists[:-1]
    k = np.argmin(gaps)
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k-1, k, k+1, k+2]
    four = dists[idx]
    raw = np.mean([max(d/D, D/d) for d in four])
    return min(1.0, 1.0 / raw) if raw > 0 else 0.0

def _polygon_score(pts):
    """Compute min vertex score."""
    n = len(pts)
    scores = [_vertex_score(pts, i) for i in range(n)]
    return min(scores)

def _build_polygon_from_angles_radii(angles, radii):
    """Build polygon from explicit angles and radii."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _score_from_angles_radii(angles, radii):
    """Score a polygon defined by angles and radii."""
    pts = _build_polygon_from_angles_radii(angles, radii)
    pts_norm, m = _normalize(pts)
    if m <= 0.01:
        return 0.0
    if not _is_strictly_convex(pts_norm):
        return 0.0
    return _polygon_score(pts_norm)

def _objective_angles(x):
    """Objective for angle optimization (radii fixed at 1)."""
    angles = x
    radii = np.ones(20)
    return -_score_from_angles_radii(angles, radii)

def _objective_radii(x):
    """Objective for radius optimization (angles fixed)."""
    radii = x
    angles = 2.0 * np.pi * np.arange(20) / 20
    return -_score_from_angles_radii(angles, radii)

def _objective_full(x):
    """Objective for full optimization: 20 angles + 20 radii."""
    angles = x[:20]
    radii = x[20:]
    return -_score_from_angles_radii(angles, radii)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized for the vertex equidistance score.
    Uses scipy differential evolution for global search, then local refinement.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Start with regular polygon
    n = 20
    base_angles = 2.0 * np.pi * np.arange(n) / n
    base_radii = np.ones(n)
    
    best_score = _score_from_angles_radii(base_angles, base_radii)
    best_angles = base_angles.copy()
    best_radii = base_radii.copy()
    
    # Phase 1: Optimize angles with radii=1 using differential evolution
    if time.time() < deadline - 200:
        try:
            bounds_angles = [(-0.3, 0.3)] * 20
            result = differential_evolution(
                _objective_angles, bounds_angles, seed=42, maxiter=100,
                popsize=15, tol=1e-8, polish=True
            )
            if not result.fun >= 0:  # If we got a positive score
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_angles = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Optimize radii with best angles using differential evolution
    if time.time() < deadline - 150:
        try:
            bounds_radii = [(0.5, 1.5)] * 20
            # Fix angles at best found
            def _obj_radii(x):
                return -_score_from_angles_radii(best_angles, x)
            
            result = differential_evolution(
                _obj_radii, bounds_radii, seed=42, maxiter=100,
                popsize=15, tol=1e-8, polish=True
            )
            if not result.fun >= 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_radii = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Joint fine-tuning with Nelder-Mead on full 40-dim
    if time.time() < deadline - 60:
        try:
            x0 = np.concatenate([best_angles, best_radii])
            result = minimize(
                _objective_full, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if not result.fun >= 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_angles = result.x[:20].copy()
                    best_radii = result.x[20:].copy()
        except Exception:
            pass
    
    # Build final polygon
    pts = _build_polygon_from_angles_radii(best_angles, best_radii)
    pts_norm, m = _normalize(pts)
    
    # Verify
    if m <= 0.01 or not _is_strictly_convex(pts_norm):
        # Fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    
    return pts_norm

# EVOLVE-BLOCK-END
