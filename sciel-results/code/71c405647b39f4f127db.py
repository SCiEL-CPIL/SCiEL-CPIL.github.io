"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0  # seconds; keep 10s safety margin under 150s kill


def _radius(pts):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    if pts.shape[0] < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    # Distance to walls
    wall = 0.5 - np.abs(pts).max()
    return min(min_pair, wall)


def _radius_fast(pts):
    """Faster radius computation using vectorized ops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    wall = 0.5 - np.abs(pts).max()
    return min(min_pair, wall)


def _neg_radius(pts):
    return -_radius_fast(pts)


def _grid_init(m, n, seed=0):
    """Create an m x n grid of points in the unit square, take first m*n >= N."""
    rng = np.random.RandomState(seed)
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n) + 0.5) / n - 0.5
    pts = np.array([(x, y) for y in ys for x in xs])
    if len(pts) > N:
        # Keep a balanced subset
        idx = np.linspace(0, len(pts) - 1, N, dtype=int)
        pts = pts[idx]
    return pts


def _random_init(seed=0):
    """Random initialization with jitter around grid."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    base = _grid_init(m, m, seed)
    base = base[:N].copy()
    # Add small random jitter
    jitter = rng.uniform(-0.02, 0.02, size=base.shape)
    base += jitter
    # Clip to valid region
    base = np.clip(base, -0.5, 0.5)
    return base


def _slsqp_optimize(pts0, max_iter=200, timeout=30):
    """Run SLSQP to maximize the minimum radius."""
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return -_radius_fast(pts)
    
    def objective_grad(x):
        pts = x.reshape(-1, 2)
        # Numerical gradient would be too slow; use finite differences
        # Actually let's just let SLSQP compute it
        return None  # Let SLSQP handle it
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        objective,
        pts0.flatten(),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12},
        callback=lambda x: None
    )
    
    elapsed = time.time() - start_time
    return result.x.reshape(-1, 2), result.fun, elapsed


def _multi_start_slsqp(deadline=140.0):
    """Multi-start SLSQP optimization."""
    start_time = time.time()
    best_pts = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Standard grid
    m = int(np.ceil(np.sqrt(N)))
    inits.append(_grid_init(m, m, seed=0))
    
    # 2. Grid with different aspect ratios
    for m, n in [(7, 8), (8, 7), (6, 9), (9, 6), (5, 10), (10, 5)]:
        if m * n >= N:
            inits.append(_grid_init(m, n, seed=1))
    
    # 3. Random inits
    for seed in range(10):
        inits.append(_random_init(seed=seed))
    
    # 4. Perturbed grid
    base_grid = _grid_init(m, m, seed=0)[:N].copy()
    for seed in range(5):
        rng = np.random.RandomState(seed + 100)
        perturbed = base_grid + rng.uniform(-0.03, 0.03, size=base_grid.shape)
        perturbed = np.clip(perturbed, -0.5, 0.5)
        inits.append(perturbed)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Evaluate initial radius
        r0 = _radius_fast(init)
        if r0 > best_radius:
            best_radius = r0
            best_pts = init.copy()
        
        # Run SLSQP with time limit
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        # Use adaptive max_iter based on remaining time
        # Estimate: ~0.1s per iteration for N=50
        max_iter = max(10, int(remaining * 5))
        
        try:
            pts_opt, fun_val, elapsed = _slsqp_optimize(init, max_iter=max_iter)
            r_opt = -fun_val
            if r_opt > best_radius:
                best_radius = r_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts if best_pts is not None else _grid_init(m, m, seed=0)[:N]


def _polish(pts, deadline=140.0):
    """Final high-precision polish pass with tighter tolerance."""
    if time.time() > deadline - 3:
        return pts
    
    try:
        bounds = [(-0.5, 0.5)] * (2 * N)
        result = minimize(
            _neg_radius,
            pts.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15}
        )
        if result.fun < -_radius_fast(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Phase 1: Multi-start SLSQP
    pts = _multi_start_slsqp(deadline=deadline)
    
    # Phase 2: Polish
    if time.time() < deadline - 5:
        pts = _polish(pts, deadline=deadline)
    
    # Final validation and clipping
    pts = np.clip(pts, -0.5, 0.5)
    return pts
# EVOLVE-BLOCK-END
