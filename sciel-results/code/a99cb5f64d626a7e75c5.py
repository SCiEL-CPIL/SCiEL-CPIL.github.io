"""Packing 50 equal circles in a unit circle: Ring init + Feasibility Binary Search + Penalty Optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

# Precompute indices for pairwise distances to speed up objective evaluation
def _get_pair_indices():
    iu = np.triu_indices(N, k=1)
    return iu[0], iu[1]

PAIR_I, PAIR_J = _get_pair_indices()

def _compute_violations(pts, r):
    """
    Compute the sum of squared violations for a given radius r.
    Violations:
    1. Pairwise distance < 2r
    2. Boundary distance < r (i.e., |c_i| + r > 1)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 1e10, np.zeros(N, dtype=int)
    if not np.all(np.isfinite(pts)):
        return 1e10, np.zeros(N, dtype=int)
    
    # Pairwise distances
    # Using precomputed indices for speed
    diff = pts[PAIR_I] - pts[PAIR_J]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    
    # Violation: dist < 2r
    # v = max(0, 2r - dist)
    pair_viol = np.maximum(0.0, 2.0 * r - dists)
    pair_viol_sq = np.sum(pair_viol**2)
    
    # Boundary
    radii = np.sqrt(np.sum(pts**2, axis=1))
    bound_viol = np.maximum(0.0, radii + r - 1.0)
    bound_viol_sq = np.sum(bound_viol**2)
    
    total_viol = pair_viol_sq + bound_viol_sq
    
    # Identify most violated pair and boundary for potential reporting, though not strictly needed for minimizer
    # We just return the scalar objective
    return total_viol, None

def _objective_penalty(x, r):
    """Objective function for scipy.optimize.minimize"""
    pts = x.reshape(N, 2)
    viol, _ = _compute_violations(pts, r)
    return viol

def _ring_to_points(radii, angles_list):
    """Convert ring parameters to point coordinates."""
    pts = []
    for i in range(len(radii)):
        r = radii[i]
        a = angles_list[i]
        if len(a) == 0:
            continue
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)

def _hexagonal_init(scale=0.12):
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x**2 + y**2 <= 0.81:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points

def _score(pts):
    """Calculate the actual packing radius score."""
    if pts is None or len(pts) != N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return 0.0
    return val

def _optimize_for_radius(pts, r, deadline):
    """
    Try to find a configuration for a given radius r.
    Returns (best_pts, best_violation).
    """
    current = pts.copy()
    
    # We want to minimize violations.
    # If violations are 0, it's feasible.
    
    # Use L-BFGS-B which is good for large scale smooth problems.
    # The objective is non-smooth due to max(0, ...), but L-BFGS-B handles it reasonably well.
    # To make it smoother, we could use a softplus approximation, but let's try direct first.
    # Actually, max(0, x)^2 is C1 continuous, so gradients exist almost everywhere.
    
    start_time = time.time()
    
    try:
        res = minimize(
            _objective_penalty,
            current.ravel(),
            args=(r,),
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-16,
                'gtol': 1e-10,
                'maxfun': 10000
            }
        )
        new_pts = res.x.reshape(N, 2)
        if not np.all(np.isfinite(new_pts)):
            return current, _compute_violations(current, r)[0]
        
        viol, _ = _compute_violations(new_pts, r)
        if viol < _compute_violations(current, r)[0]:
            return new_pts, viol
        else:
            return current, _compute_violations(current, r)[0]
            
    except Exception:
        return current, _compute_violations(current, r)[0]

def _binary_search_radius(pts, deadline):
    """
    Binary search for the maximum radius r such that a feasible configuration exists.
    """
    # Initial score
    r_low = _score(pts)
    if r_low <= 0:
        return pts, r_low
    
    # Set a high bound. The theoretical max for N=50 is around 0.126.
    # Let's start with a bound slightly above current best.
    r_high = r_low + 0.005
    
    best_pts = pts.copy()
    best_r = r_low
    
    # We will iterate to find the max r.
    # We do a binary search on r.
    for _ in range(20): # ~20 iterations is enough for precision
        if time.time() > deadline - 5:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # We start from the best known feasible configuration (best_pts)
        # or the current best_pts from the previous step.
        candidate_pts, viol = _optimize_for_radius(best_pts, r_mid, deadline)
        
        # Check if it's actually feasible
        # A violation of < 1e-8 is considered feasible for double precision
        if viol < 1e-8:
            # Feasible! We can try a larger r.
            r_low = r_mid
            best_r = r_mid
            best_pts = candidate_pts.copy()
        else:
            # Not feasible. Try a smaller r.
            r_high = r_mid
            
    return best_pts, best_r

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_points = None
    best_score = -1.0
    
    # Ring topologies for 50 circles
    # These are based on known good packings
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 13, 13, 17],
        [1, 6, 12, 12, 19],
        [1, 6, 11, 13, 19],
    ]
    
    candidates = []
    
    # Generate ring-based initializations
    for counts in ring_structures:
        n_rings = len(counts)
        # Estimate radii. Total area ~ N * pi * r^2.
        # If r ~ 0.12, area ~ 50 * 0.045 ~ 2.25. Unit circle area pi ~ 3.14.
        # Radii should be roughly proportional to sqrt(k) for k-th ring?
        # Let's just use linear spacing for now, or better, optimize radii too?
        # For simplicity, fix radii to reasonable values.
        # Center is 0.
        # Outer ring radius should be close to 1 - r.
        # Let's interpolate.
        
        # A better heuristic for radii:
        # r_0 = 0
        # r_k = (1 - r_est) * sqrt(k / K) ? No, that's for area filling.
        # Let's just use fixed fractions of the max radius.
        # Max radius approx 0.88 (1 - 0.12)
        max_r = 0.88
        radii0 = np.linspace(0, max_r, n_rings)
        radii0[0] = 0.0
        
        angles_list = []
        for i, n_pts in enumerate(counts):
            if n_pts == 1:
                angles_list.append(np.array([0.0]))
            else:
                # Offset angles for inner rings to avoid alignment?
                offset = (i % 2) * (np.pi / n_pts)
                angles_list.append(np.linspace(offset, 2 * np.pi + offset, n_pts, endpoint=False))
        
        pts = _ring_to_points(radii0, angles_list)
        s = _score(pts)
        if s > 0:
            candidates.append((pts.copy(), s))
    
    # Hexagonal initialization
    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hex_score = _score(hex_pts)
        if hex_score > 0:
            candidates.append((hex_pts.copy(), hex_score))
    
    # Sort candidates by initial score (best first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Process top candidates
    # We have ~135s.
    # Binary search takes ~20 * (1 L-BFGS run).
    # L-BFGS run takes ~1-2s?
    # So 20 * 2 = 40s per candidate.
    # We can afford 2-3 candidates.
    
    max_candidates_to_try = 3
    for idx in range(max_candidates_to_try):
        if time.time() > deadline - 30:
            break
        if idx >= len(candidates):
            break
            
        pts, s = candidates[idx]
        
        # Run binary search polish
        polished_pts, polished_score = _binary_search_radius(pts, deadline)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
            
    # If no good candidates, use hex
    if best_points is None:
        best_points = hex_pts.copy()
        best_score = _score(best_points)
    
    # Final check and clamp
    if best_points is not None:
        r_final = _score(best_points)
        if r_final > 0:
            r2 = np.hypot(best_points[:, 0], best_points[:, 1])
            for i in range(N):
                if r2[i] > 1 - r_final + 1e-10:
                    scale = (1 - r_final) / r2[i]
                    best_points[i] *= scale
    
    if best_points is None:
        # Fallback: just return a valid grid
        best_points = _hexagonal_init(0.1)
        if best_points is None:
            best_points = np.zeros((N, 2))
            
    return best_points.reshape(N, 2)
