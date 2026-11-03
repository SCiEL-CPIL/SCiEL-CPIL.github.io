"""Packing 50 equal circles in unit square via L-BFGS-B with soft-min radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _soft_min_radius(pts, beta):
    """Compute soft-min of pairwise distances and wall distances."""
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0, None] - pts[None, :, 0]
    dy = pts[:, 1, None] - pts[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    # Mask out self-distances
    np.fill_diagonal(d, np.inf)
    pair_min = d.min()
    # Wall distances: min over each point of distance to each wall
    # Square is [-0.5, 0.5]^2
    wall_dists = np.minimum.reduce([
        pts[:, 0] + 0.5,  # distance to left wall
        0.5 - pts[:, 0],  # distance to right wall
        pts[:, 1] + 0.5,  # distance to bottom wall
        0.5 - pts[:, 1],  # distance to top wall
    ])
    wall_min = wall_dists.min()
    # Soft-min of all distances (pair + wall)
    all_dists = np.concatenate([d.ravel(), wall_dists.ravel()])
    # Remove inf values
    all_dists = all_dists[all_dists < 1e10]
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    min_d = all_dists.min()
    # Shift for numerical stability
    exp_terms = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_terms.sum()) / beta
    return soft_min

def _objective_and_grad(pts_flat, beta):
    pts = pts_flat.reshape(-1, 2)
    r = _soft_min_radius(pts, beta)
    # We want to maximize r, so minimize -r
    obj = -r
    # Gradient via finite differences would be too slow; use analytical
    # Actually, let's just use finite differences with a reasonable step
    # But that's O(N) evaluations per gradient which is expensive.
    # Let's use scipy's numerical gradient by not providing jac
    return obj

def construct():
    deadline = time.time() + 115.0  # 120s limit, 5s safety margin
    
    # Multiple random initializations
    best_score = -np.inf
    best_pts = None
    
    # Start with a grid-like initialization for stability
    m = 7  # 7x7 grid has 49 cells, we need 50
    # Actually, use a slightly perturbed grid
    xs = np.linspace(-0.45, 0.45, 7)
    ys = np.linspace(-0.45, 0.45, 7)
    grid_pts = np.array([[x, y] for y in ys for x in xs])
    
    # We have 49 grid points, add one more
    # Place the 50th at a random position
    rng = np.random.default_rng(42)
    
    inits = []
    # Grid-based init
    grid_50 = np.vstack([grid_pts[:49], [0.0, 0.0]])  # placeholder
    inits.append(grid_50.copy())
    
    # Random inits
    for _ in range(5):
        pts = rng.uniform(-0.45, 0.45, size=(50, 2))
        inits.append(pts)
    
    # Also try a hexagonal-like packing
    # Hexagonal packing in a 1x1 square
    hex_pts = []
    spacing = 0.15
    y = -0.45
    row = 0
    while len(hex_pts) < 50 and y < 0.45:
        x_offset = 0 if row % 2 == 0 else spacing * 0.866
        x = -0.45 + x_offset
        while x < 0.45 and len(hex_pts) < 50:
            hex_pts.append([x, y])
            x += spacing
        y += spacing * 0.866
        row += 1
    if len(hex_pts) >= 50:
        inits.append(np.array(hex_pts[:50]))
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Progressive beta schedule
        betas = [50, 100, 200, 500, 1000]
        
        x0 = init.ravel()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def obj(x):
                return _objective_and_grad(x, beta)
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * 100,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success or result.fun < best_score * -1:
                    x0 = result.x
            except Exception:
                pass
        
        # Compute actual score for this configuration
        pts = x0.reshape(-1, 2)
        # Check validity
        if np.any(~np.isfinite(pts)):
            continue
        if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
           np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
            continue
        
        # Compute true min radius
        dx = pts[:, 0, None] - pts[None, :, 0]
        dy = pts[:, 1, None] - pts[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        pair_min = d.min() / 2
        
        wall_dists = np.minimum.reduce([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1],
        ])
        wall_min = wall_dists.min()
        
        r = min(pair_min, wall_min)
        
        if r > best_score:
            best_score = r
            best_pts = pts.copy()
    
    # Final polish pass with high beta if time permits
    if best_pts is not None and time.time() < deadline - 5:
        x0 = best_pts.ravel()
        for beta in [2000, 5000]:
            if time.time() > deadline - 2:
                break
            def obj(x):
                return _objective_and_grad(x, beta)
            try:
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * 100,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x0 = result.x
            except Exception:
                pass
        
        pts = x0.reshape(-1, 2)
        if np.all(np.isfinite(pts)) and \
           np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
           np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
            dx = pts[:, 0, None] - pts[None, :, 0]
            dy = pts[:, 1, None] - pts[None, :, 1]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            pair_min = d.min() / 2
            wall_dists = np.minimum.reduce([
                pts[:, 0] + 0.5,
                0.5 - pts[:, 0],
                pts[:, 1] + 0.5,
                0.5 - pts[:, 1],
            ])
            wall_min = wall_dists.min()
            r = min(pair_min, wall_min)
            if r > best_score:
                best_score = r
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: simple grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
