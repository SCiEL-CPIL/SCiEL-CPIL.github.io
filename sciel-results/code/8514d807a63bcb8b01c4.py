"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                    (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _soft_min_triangle_area(pts, beta):
    """Soft-min of triangle areas using log-sum-exp (negative for min)."""
    n = pts.shape[0]
    areas = np.empty((n, n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs(
                    (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                    (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                )
                areas[i, j, k] = a
    # Only consider valid triples (upper triangle)
    mask = np.zeros((n, n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask[i, j, k] = True
    vals = areas[mask]
    # Soft min: -log(sum(exp(-beta * a))) / beta
    shifted = -beta * vals
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta


def _objective(params, beta):
    """
    Parameters: 14 points (28 coords) + 3 affine (rotation angle, scale, shear)
    Actually, let's use a simpler parameterization:
    - 14 points as vertices of a regular heptagon, each displaced radially
    - Plus global affine: rotation, scale, shear
    Total: 7 radial + 3 affine = 10 params
    
    But let's also try direct 28D coordinate optimization.
    """
    # Direct 28D: optimize all coordinates
    pts = params.reshape((14, 2))
    # Soft min triangle area (we want to maximize min area, so minimize negative)
    # But we also need to normalize by hull area
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    soft_min_ta = _soft_min_triangle_area(pts, beta)
    # We want to maximize min_ta / hull_area
    # Equivalent to minimizing -min_ta / hull_area
    # Using soft min as surrogate:
    return -soft_min_ta / hull_area


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_score = 0.0
    best_pts = None
    
    # Strategy 1: Regular heptagon with radial displacements + affine
    # 7 vertices of regular heptagon, each can move radially
    # Plus we duplicate/interleave to get 14 points? 
    # Actually, let's use 14 points on a circle with angular perturbations
    # and radial perturbations, plus affine transform
    
    # Better approach: Start from a regular 14-gon and optimize all coordinates
    # with soft-min objective and beta annealing
    
    # Initial: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_params = init_pts.flatten()
    
    # Compute initial score
    init_score = _score(init_pts)
    best_score = init_score
    best_pts = init_pts.copy()
    
    # Beta annealing schedule
    betas = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    current_params = init_params.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Try L-BFGS-B optimization
        result = minimize(
            _objective,
            current_params,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < 1e5:
            candidate_pts = result.fun  # placeholder
            cand_params = result.x
            cand_pts = cand_params.reshape((14, 2))
            cand_score = _score(cand_pts)
            
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts.copy()
                current_params = cand_params.copy()
    
    # Multi-start: try different initial configurations
    rng = np.random.RandomState(42)
    num_starts = 5
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        if s == 0:
            # Regular 14-gon
            init_p = init_params.copy()
        elif s == 1:
            # Regular 14-gon with small random perturbations
            init_p = init_params + rng.randn(28) * 0.1
        elif s == 2:
            # Two concentric heptagons
            angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            pts_inner = 0.5 * np.column_stack([np.cos(angles7), np.sin(angles7)])
            pts_outer = np.column_stack([np.cos(angles7 + np.pi/7), np.sin(angles7 + np.pi/7)])
            init_p = np.vstack([pts_inner, pts_outer]).flatten()
        elif s == 3:
            # Random points in unit circle
            r = np.sqrt(rng.rand(14))
            theta = rng.rand(14) * 2 * np.pi
            init_p = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()
        else:
            # Regular 14-gon rotated
            angle_offset = s * 0.1
            angles_rot = angles + angle_offset
            init_p = np.column_stack([np.cos(angles_rot), np.sin(angles_rot)]).flatten()
        
        # Run beta annealing from this start
        curr = init_p.copy()
        for beta in [5.0, 50.0, 200.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective,
                    curr,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 3000, 'gtol': 1e-5}
                )
                if res.fun < 1e5:
                    cand_pts = res.x.reshape((14, 2))
                    cand_score = _score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts.copy()
                        curr = res.x.copy()
            except Exception:
                pass
    
    # Final polish: high-beta local optimization from best point
    if best_pts is not None and time.time() < deadline - 10:
        for beta in [1000.0, 5000.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _objective,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 10000, 'gtol': 1e-8}
                )
                if res.fun < 1e5:
                    cand_pts = res.x.reshape((14, 2))
                    cand_score = _score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts.copy()
            except Exception:
                pass
    
    return best_pts


if __name__ == '__main__':
    pts = construct()
    print(f"Score: {_score(pts)}")
