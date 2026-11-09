"""Fifty equal circles in a square – high-precision packing optimizer."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # Safety margin before 120s kill


def _score_from_centres(c):
    """Compute the exact radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    
    # Pairwise distances
    # Using pdist for efficiency
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    
    # Distance to walls
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -t + penalty for violations.
    x = [c_1, ..., c_N, t] where c_i are 2D centers and t is the radius.
    
    We want to maximize t such that:
    - All pairwise distances >= 2t
    - All wall distances >= t
    
    This is equivalent to minimizing -t subject to constraints.
    We use a smooth penalty for constraint violations.
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Clip centers to stay in bounds
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances / 2
    d = pdist(c) / 2.0
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    
    # All constraints: dist_i >= t
    all_dists = np.concatenate([d, wall_dists_flat])
    
    # Violations: max(0, t - dist_i)
    violations = np.maximum(0.0, t - all_dists)
    
    # Smooth penalty: sum of exp(-beta * violation) for violated constraints
    # This is 0 when no violations, large when violations exist
    if violations.size > 0:
        max_v = violations.max()
        if max_v > 1e-15:
            # Use soft penalty
            penalty = np.exp(-beta * violations).sum() / beta
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    return -t + penalty


def _negative_smooth_score(c_flat, beta=100.0):
    """Objective to minimise (legacy, kept for compatibility)."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    all_dists = np.concatenate([d, wall_dists_flat])
    
    min_d = all_dists.min()
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    
    return -soft_min


def _grid_init(rows, cols):
    """Generate a grid initial layout."""
    # Place N circles in a grid of rows x cols
    # We need to fit N into rows*cols.
    # For N=50, 7x7=49 is too small. 7x8=56.
    # We will use a 7x8 grid and remove 6 points, or just use 8x7?
    # Better: Just use a grid that covers the square.
    
    # Use aspect ratio to determine rows/cols
    # For square, rows ~= cols
    # 7x8 grid:
    x = np.linspace(-HALF, HALF, cols)
    y = np.linspace(-HALF, HALF, rows)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # If we have more points than N, subsample
    if len(grid) > N:
        # Remove points from the edge or random?
        # Just take the first N? Or center them?
        # Centering is better.
        # We generated a full grid. Let's just take a subset.
        # To be fair, let's re-generate exactly N points in a grid-like manner.
        pass
        
    # Simple robust init:
    # Map indices to grid positions
    pts = []
    # Try to find best R, C such that R*C >= N
    best_r, best_c = 7, 8
    for r in range(5, 10):
        for c in range(5, 10):
            if r * c >= N:
                # Prefer square-ish
                if abs(r-c) < abs(best_r-best_c):
                    best_r, best_c = r, c
    
    x = np.linspace(-HALF, HALF, best_c)
    y = np.linspace(-HALF, HALF, best_r)
    
    # Create full grid
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # Shuffle and take N? No, that breaks grid structure.
    # Remove points from the corners/edges to maintain symmetry?
    # Or just take the first N in a space-filling curve?
    
    # Simplest: Take the first N points of the grid (top-left to bottom-right)
    # This might bias the packing.
    # Better: Randomly select N points from the grid?
    # Or, generate a grid of exactly N points by adjusting spacing?
    
    # Let's just use the 7x8 grid and drop 6 points randomly from the perimeter
    # to keep the core intact.
    idx = np.arange(len(full_grid))
    # Prioritize keeping center
    # Distance from center
    dists_to_center = np.sum(full_grid**2, axis=1)
    # Sort by distance, keep closest N
    sorted_idx = np.argsort(dists_to_center)
    pts = full_grid[sorted_idx[:N]]
    return pts


def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    # Hexagonal close packing
    # Row spacing: r * sqrt(3)
    # Col spacing: 2r
    # Let's use a fixed r for init, say 0.07
    r_init = 0.07
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    # Determine grid size
    max_x = HALF - r_init
    max_y = HALF - r_init
    
    row = 0
    y = -max_y
    while y <= max_y:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 2: # Safety break
            break
            
    pts = np.array(points)
    # Shuffle and take N?
    # Or take closest to center?
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts


def _random_init(rng):
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3):
    """
    Multi-stage optimization using epigraph formulation.
    Returns optimized centers.
    """
    # Initial t estimate
    c0 = x0.reshape(N, 2)
    r0 = _score_from_centres(c0)
    t0_init = max(r0, 0.01)
    
    x = np.concatenate([c0.ravel(), [t0_init]])
    
    # Bounds for centers
    bounds_c = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    # Bounds for t
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    # Multi-stage with increasing beta
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        try:
            res = minimize(
                lambda x_: _epigraph_objective(x_, beta=beta),
                x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
            )
            x = res.x
        except Exception:
            pass
    
    # Extract centers
    c = x[:2*N].reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    return c


def _polish_powell(c, timeout=10.0):
    """
    Final high-precision polish using Powell's method to maximize the minimum distance.
    This handles the non-smooth min function and is often faster than Nelder-Mead in high dimensions.
    """
    t_start = time.time()
    
    def objective(x):
        c_local = x.reshape(N, 2)
        c_local = np.clip(c_local, -HALF + 1e-12, HALF - 1e-12)
        # We want to maximize r, so minimize -r
        d = pdist(c_local)
        min_pair = d.min() if d.size > 0 else 0.0
        
        dist_walls = np.min(np.stack([
            c_local[:, 0] + HALF,
            HALF - c_local[:, 0],
            c_local[:, 1] + HALF,
            HALF - c_local[:, 1]
        ]), axis=0)
        min_wall = dist_walls.min()
        
        r = min(min_pair / 2.0, min_wall)
        return -r

    x0 = c.ravel()
    
    # Powell options
    options = {
        'maxiter': 500,
        'xtol': 1e-12,
        'ftol': 1e-12,
        'disp': False
    }
    
    try:
        res = minimize(
            objective,
            x0,
            method='Powell',
            options=options
        )
        c_opt = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        
        # Verify it's better
        s_opt = _score_from_centres(c_opt)
        s_init = _score_from_centres(c)
        
        if s_opt > s_init:
            return c_opt
        else:
            return c
            
    except Exception:
        return c

def _polish_high_precision(c, max_iter=1000):
    """
    Final high-precision polish using very high beta soft-min.
    """
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    # Use progressively higher beta
    for beta in [1000.0, 2000.0, 5000.0]:
        try:
            res = minimize(
                lambda cf: _negative_smooth_score(cf, beta=beta),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16, 'iprint': 0}
            )
            x0 = res.x
        except Exception:
            pass
    
    c = np.clip(x0.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    return c


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial configurations
    inits = [
        _grid_init(7, 8),
        _hex_init(),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    # Optimization loop with epigraph formulation
    n_starts = 25
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 20:
            break
            
        if i < len(inits):
            x0 = inits[i].ravel()
        else:
            # Perturb best or random
            if i % 3 == 0:
                # Small perturbation of best
                x0 = best_centres.ravel() + rng.normal(0, 0.01, size=2*N)
            elif i % 3 == 1:
                # Medium perturbation
                x0 = best_centres.ravel() + rng.normal(0, 0.03, size=2*N)
            else:
                # Random start
                x0 = _random_init(rng).ravel()
                
        x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
        
        try:
            # Step 1: Epigraph optimization
            c_opt = _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3)
            
            # Step 2: High-precision polish if time permits
            if time.time() - t0 < DEADLINE - 10:
                c_opt = _polish_high_precision(c_opt, max_iter=500)
            
            s = _score_from_centres(c_opt)
            if s > best_score:
                best_score = s
                best_centres = c_opt
                
        except Exception:
            pass

    # Final high-precision polish on best result
    if time.time() - t0 < DEADLINE - 5:
        try:
            # First, standard L-BFGS polish
            best_centres = _polish_high_precision(best_centres, max_iter=500)
            final_score = _score_from_centres(best_centres)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass

    # Second, Powell polish for non-smooth refinement
    if time.time() - t0 < DEADLINE - 2:
        try:
            # Powell is generally faster and more robust than NM in high dimensions
            if time.time() - t0 < DEADLINE - 10:
                best_centres = _polish_powell(best_centres)
                final_score = _score_from_centres(best_centres)
                if final_score > best_score:
                    best_score = final_score
        except Exception:
            pass

    # Final validation
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
