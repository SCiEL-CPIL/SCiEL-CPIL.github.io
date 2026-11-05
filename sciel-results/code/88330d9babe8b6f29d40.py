"""Optimized 8-point configuration maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _score(points):
    """Compute Delta / n^n where Delta is product over ordered pairs of distances
    after scaling to diameter 2. The regular octagon scores exactly 1."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Diameter is the max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    dists_scaled = dists / diam * 2.0
    # Product over ordered pairs (i != j)
    # Use log to avoid overflow
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = math.exp(log_prod - n * math.log(n))
    return score

def _negative_score(params):
    """Negative score for minimization. params is a flat array of 16 floats (8 points)."""
    pts = params.reshape(8, 2)
    # Check for near-duplicate points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    if np.min(dists[mask]) < 1e-8:
        return 1e10
    s = _score(pts)
    return -s

def _regular_octagon():
    return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]

def _two_square_config(theta):
    """Two squares rotated by theta, centered at origin.
    Square 1: vertices at angles 0, pi/2, pi, 3pi/2 with radius r1
    Square 2: vertices at angles theta, theta+pi/2, theta+pi, theta+3pi/2 with radius r2
    We optimize theta and the radii ratio.
    Actually, let's parametrize more generally:
    4 points on a circle of radius r1 at angles a1, a1+pi/2, a1+pi, a1+3pi/2
    4 points on a circle of radius r2 at angles a2, a2+pi/2, a2+pi, a2+3pi/2
    But for symmetry, we can fix a1=0 and optimize a2, r1, r2.
    """
    pts = []
    for k in range(4):
        ang = k * math.pi / 2
        pts.append([math.cos(ang), math.sin(ang)])
    for k in range(4):
        ang = theta + k * math.pi / 2
        pts.append([math.cos(ang), math.sin(ang)])
    return pts

def _parametric_two_squares(theta, r1, r2):
    """Two squares: first with radius r1, second with radius r2, rotated by theta."""
    pts = []
    for k in range(4):
        ang = k * math.pi / 2
        pts.append([r1 * math.cos(ang), r1 * math.sin(ang)])
    for k in range(4):
        ang = theta + k * math.pi / 2
        pts.append([r2 * math.cos(ang), r2 * math.sin(ang)])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with regular octagon
    best_pts = _regular_octagon()
    best_score = _score(best_pts)
    
    # Phase 1: Grid search over two-square parameterization
    # Parameters: theta (rotation of second square), r1 (radius of first square), r2 (radius of second square)
    # We can fix the overall scale since the score is scale-invariant (after diameter normalization).
    # So we can fix r1 = 1 and optimize theta and r2.
    
    theta_vals = np.linspace(0, math.pi / 4, 41)  # By symmetry, theta in [0, pi/4]
    r2_vals = np.linspace(0.5, 1.5, 41)
    
    for theta in theta_vals:
        if time.time() > deadline:
            break
        for r2 in r2_vals:
            pts = _parametric_two_squares(theta, 1.0, r2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    # Phase 2: Refine with L-BFGS-B
    from scipy.optimize import minimize
    
    # Convert best_pts to flat params for optimization
    # We'll optimize all 16 coordinates directly
    x0 = np.array(best_pts, dtype=np.float64).flatten()
    
    if time.time() < deadline - 10:
        result = minimize(_negative_score, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success and -result.fun > best_score:
            best_score = -result.fun
            best_pts = result.x.reshape(8, 2).tolist()
    
    # Phase 3: Try multiple random restarts with the two-square parameterization
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        for trial in range(20):
            if time.time() > deadline:
                break
            theta0 = rng.uniform(0, math.pi / 4)
            r1_0 = rng.uniform(0.7, 1.3)
            r2_0 = rng.uniform(0.7, 1.3)
            x0 = np.array(_parametric_two_squares(theta0, r1_0, r2_0), dtype=np.float64).flatten()
            
            result = minimize(_negative_score, x0, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success:
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_pts = result.x.reshape(8, 2).tolist()
    
    # Phase 4: Final polish with full 16D optimization from the best point
    if time.time() < deadline - 5:
        x0 = np.array(best_pts, dtype=np.float64).flatten()
        result = minimize(_negative_score, x0, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success and -result.fun > best_score:
            best_score = -result.fun
            best_pts = result.x.reshape(8, 2).tolist()
    
    # Ensure all points are distinct and coordinates are within bounds
    pts = np.array(best_pts, dtype=np.float64)
    # Check distinctness
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    if np.min(dists[mask]) < 1e-6:
        # Fall back to regular octagon
        best_pts = _regular_octagon()
    
    # Clip coordinates to be within bounds
    best_pts = np.clip(pts, -1e6, 1e6).tolist()
    
    return best_pts
# EVOLVE-BLOCK-END
