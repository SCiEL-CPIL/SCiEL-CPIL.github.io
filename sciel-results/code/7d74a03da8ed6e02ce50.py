import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14
DEADLINE = 140  # Safety margin

def _min_triangle_area(pts):
    """Return the minimum triangle area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized computation for speed
    # We can compute all cross products at once?
    # For N=14, a triple loop is 364 iterations, which is very fast.
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-15:
                        return 0.0
    return min_area

def _hull_area(pts):
    """Return the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    if points is None:
        return -1.0
    pts = np.asarray(points, dtype=float)
    if pts.shape != (N, 2):
        return -1.0
    ha = _hull_area(pts)
    if ha < 1e-15:
        return -1.0
    mt = _min_triangle_area(pts)
    return mt / ha

def _regular_polygon():
    """Regular N-gon inscribed in unit circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _two_ring_config(r, theta):
    """
    Generate 14 points: 7 on unit circle, 7 on circle of radius r,
    rotated by theta relative to the outer ring.
    """
    n_outer = 7
    n_inner = 7
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = angles_outer + theta
    
    pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    pts_inner = r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    
    pts = np.vstack((pts_outer, pts_inner))
    return pts

def _neg_score_two_ring(params):
    """Objective function for differential evolution on 2-ring parameters."""
    r, theta = params
    if r < 0 or r > 1:
        return 1.0  # Penalty
    pts = _two_ring_config(r, theta)
    sc = _score(pts)
    return -sc

def construct():
    start = time.time()
    
    # Baseline: Regular 14-gon
    best_pts = _regular_polygon()
    best_score = _score(best_pts)
    
    # Search 2-ring configurations
    # r: radius of inner ring [0, 1]
    # theta: angular offset [0, 2*pi/7] (symmetry allows restricting to one sector)
    bounds = [(0.0, 1.0), (0.0, 2 * np.pi / 7)]
    
    try:
        result = differential_evolution(
            _neg_score_two_ring,
            bounds,
            maxiter=200,
            tol=1e-8,
            seed=42,
            polish=True,
            popsize=20,
            init='sobol'
        )
        
        elapsed = time.time() - start
        if elapsed < DEADLINE:
            r_opt, theta_opt = result.x
            cand_pts = _two_ring_config(r_opt, theta_opt)
            cand_score = _score(cand_pts)
            
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
                
    except Exception:
        pass

    # Local polish in full 28D space around the best point found
    # This helps if the symmetric assumption is slightly off
    elapsed = time.time() - start
    if elapsed < DEADLINE - 5:
        # Use Nelder-Mead or simple coordinate ascent?
        # Simple random perturbation is safer and fast enough for small N
        step = 0.05
        for _ in range(200):
            if time.time() - start > DEADLINE - 2:
                break
            pts = best_pts + step * np.random.randn(N, 2)
            # Note: We don't clip to [0,1] because the problem is scale/translation invariant
            # and we just want to explore the local neighborhood
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts
            step *= 0.9 # Decay step size

    return best_pts
