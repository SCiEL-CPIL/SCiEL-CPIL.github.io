"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import time

N = 200
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _jittered_hex_init(rows, cols, jitter_scale=0.01):
    """Generate a jittered hexagonal lattice initialization."""
    dx = 1.0 / cols
    dy = dx * np.sqrt(3) / 2
    # Adjust dy to fit in the square
    total_height = (rows - 1) * dy
    if total_height > 1.0:
        dy = 1.0 / (rows - 1)
    
    x0 = -0.5 + dx / 2
    y0 = -0.5 + dy / 2
    
    pts = []
    for i in range(rows):
        y = y0 + i * dy
        shift = dx * 0.5 if i % 2 == 1 else 0.0
        xs = x0 + shift + np.arange(cols) * dx
        for x in xs:
            pts.append([x, y])
    
    pts = np.array(pts[:N])
    if len(pts) < N:
        return _best_result()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    pts += rng.uniform(-jitter_scale, jitter_scale, size=pts.shape)
    
    # Clamp to square
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts


def _soft_min_radius(pts, temperature=0.01):
    """Soft-min approximation of the radius using log-sum-exp.
    
    The radius is min(min_pair_dist/2, min_boundary_dist).
    We approximate min(a, b) using -log(exp(-a/T) + exp(-b/T)) * T
    for small T, which is smooth and differentiable.
    """
    pts = pts.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return -1e10
    
    # Compute pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    # Compute boundary distances
    dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Soft min
    a = min_pair / 2.0
    b = min_boundary
    
    # Use log-sum-exp for numerical stability
    m = min(a, b)
    soft_min = m - temperature * np.log(np.exp((a - m) / temperature) + np.exp((b - m) / temperature))
    
    return soft_min


def _penalty_objective(params, temperature=0.01, penalty_weight=100.0):
    """Penalty-based objective: maximize soft-min radius with boundary penalties."""
    pts = params.reshape(-1, 2)
    
    # Boundary penalty: penalize points outside [-0.5, 0.5]
    outside = np.maximum(0, np.abs(pts) - 0.5)
    boundary_penalty = penalty_weight * np.sum(outside ** 2)
    
    # Soft-min radius (we want to maximize, so negate)
    radius = _soft_min_radius(pts, temperature)
    
    # Total objective: -radius + penalty
    return -radius + boundary_penalty


def _neg_rect_score(params, rows, cols):
    dx, dy, x0, y0 = params
    if dx <= 0 or dy <= 0:
        return 1e10
    xs = x0 + np.arange(cols) * dx
    ys = y0 + np.arange(rows) * dy
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    if len(pts) < N:
        return 1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    if N > 1:
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
    else:
        min_pair = np.inf
    r = min(wall, min_pair / 2)
    if r <= 0:
        return 1e10
    return -r


def _local_refine(points, time_budget):
    """Local refinement using penalty-based L-BFGS-B optimization."""
    start = time.time()
    current = points.copy()
    current_score = _score(current)
    
    # Phase 1: Coarse optimization with larger temperature
    try:
        res = minimize(_penalty_objective, current.flatten(), 
                      method='L-BFGS-B',
                      bounds=[(-0.5, 0.5)] * (2 * N),
                      args=(0.05, 100.0),
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        
        elapsed = time.time() - start
        if elapsed < time_budget * 0.6:
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
    except Exception:
        pass
    
    # Phase 2: Fine optimization with smaller temperature
    remaining = time_budget - (time.time() - start)
    if remaining > 10:
        try:
            res2 = minimize(_penalty_objective, current.flatten(),
                           method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           args=(0.005, 500.0),
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            new_pts = res2.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Phase 3: Final polish with very small temperature
    remaining = time_budget - (time.time() - start)
    if remaining > 5:
        try:
            res3 = minimize(_penalty_objective, current.flatten(),
                           method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           args=(0.001, 1000.0),
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            new_pts = res3.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current, current_score


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = _best_result()
    best_r = _score(best_pts)

    # Try different hexagonal grid configurations
    factor_pairs = []
    for c in range(1, int(np.sqrt(N)) + 1):
        if N % c == 0:
            r = N // c
            factor_pairs.append((r, c))
            factor_pairs.append((c, r))
    factor_pairs = list(set(factor_pairs))

    extra = [(14, 15), (15, 14), (20, 10), (10, 20), (25, 8), (8, 25),
             (16, 13), (13, 16), (12, 17), (17, 12), (20, 10), (10, 20)]
    for pair in extra:
        if pair not in factor_pairs:
            factor_pairs.append(pair)

    # Try jittered hexagonal initializations with penalty optimization
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT - 30:
            break
        
        init_pts = _jittered_hex_init(rows, cols, jitter_scale=0.005)
        init_score = _score(init_pts)
        
        if init_score > best_r:
            best_r = init_score
            best_pts = init_pts.copy()
        
        # Refine with penalty-based optimization
        remaining = _TIME_LIMIT - (time.time() - t0) - 15
        if remaining > 5:
            refined_pts, refined_score = _local_refine(init_pts, remaining)
            if refined_score > best_r:
                best_r = refined_score
                best_pts = refined_pts.copy()

    # Final refinement of the best configuration
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 10:
        best_pts, best_r = _local_refine(best_pts, remaining - 5)

    if best_r <= 0:
        return _best_result()
    return best_pts
