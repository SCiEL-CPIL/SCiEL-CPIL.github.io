"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_value(circles):
    """Compute sum of radii and check validity. Returns (value, valid)."""
    if circles is None or len(circles) != N:
        return 0.0, False
    if not np.all(np.isfinite(circles)):
        return 0.0, False
    radii = circles[:, 2]
    if np.any(radii <= 0):
        return 0.0, False
    if np.any(radii > 1.0):
        return 0.0, False
    
    # Bounding box
    min_x = np.min(circles[:, 0] - radii)
    max_x = np.max(circles[:, 0] + radii)
    min_y = np.min(circles[:, 1] - radii)
    max_y = np.max(circles[:, 1] + radii)
    
    W = max_x - min_x
    H = max_y - min_y
    
    if W <= 0 or H <= 0:
        return 0.0, False
    if 2 * (W + H) > 4.0 + 1e-9:
        return 0.0, False
    
    # Check non-overlap: distance >= r_i + r_j for all pairs
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = radii
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    
    return np.sum(radii), True


def _objective(params, W, H):
    """
    Objective to minimize: negative sum of radii plus penalty for overlaps and boundary violations.
    params: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{N-1}, y_{N-1}, r_{N-1}]
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Penalty for negative radii
    pen = 0.0
    if np.any(rs < 0):
        pen += 100.0 * np.sum(np.minimum(rs, 0)**2)
    
    # Boundary penalties
    # Circles must be within [0, W] x [0, H]
    bx_pen = 0.0
    for i in range(n):
        if xs[i] - rs[i] < 0:
            bx_pen += (xs[i] - rs[i])**2
        if xs[i] + rs[i] > W:
            bx_pen += (xs[i] + rs[i] - W)**2
        if ys[i] - rs[i] < 0:
            bx_pen += (ys[i] - rs[i])**2
        if ys[i] + rs[i] > H:
            bx_pen += (ys[i] + rs[i] - H)**2
    
    # Overlap penalties
    ov_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                ov_pen += (min_dist - dist)**2
    
    total_penalty = 100.0 * (bx_pen + ov_pen)
    
    return -np.sum(rs) + total_penalty


def _make_grid_initial(W, H, n_cols, n_rows):
    """Create initial grid of circles."""
    circles = []
    for i in range(n_cols):
        for j in range(n_rows):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / n_cols
            cy = (j + 0.5) * H / n_rows
            r = min(W / n_cols, H / n_rows) * 0.49
            circles.append([cx, cy, r])
    return np.array(circles[:N])


def _optimize_for_aspect_ratio(W, H, n_cols, n_rows, time_limit):
    """Run optimization for a given aspect ratio and grid config."""
    start = time.time()
    
    # Initial guess
    circles = _make_grid_initial(W, H, n_cols, n_rows)
    params = circles.flatten()
    
    best_params = params.copy()
    best_obj = _objective(params, W, H)
    
    # Try multiple random restarts
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit - 5:
            break
        
        if restart == 0:
            p = params.copy()
        else:
            p = params.copy()
            # Perturb
            p[0::3] += rng.normal(0, W*0.1, N)
            p[1::3] += rng.normal(0, H*0.1, N)
            p[2::3] *= (1 + rng.normal(0, 0.2, N))
            p[2::3] = np.clip(p[2::3], 0.01, min(W, H)/2)
        
        res = minimize(_objective, p, args=(W, H), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    # Extract and validate
    circles = best_params.reshape(N, 3)
    # Clamp to valid range
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
    circles[:, 2] = np.clip(circles[:, 2], 1e-6, min(W, H)/2)
    
    val, valid = _compute_value(circles)
    if valid:
        return val, circles
    return 0.0, None


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_val = 0.0
    best_circles = None
    
    # Try different aspect ratios and grid configurations
    # Perimeter = 2(W+H) = 4 => W+H = 2
    aspect_ratios = [
        (1.0, 1.0),       # square
        (1.5, 0.5),       # wide
        (0.5, 1.5),       # tall
        (2.0, 0.0),       # degenerate, skip
        (1.2, 0.8),
        (0.8, 1.2),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.1, 0.9),
        (0.9, 1.1),
    ]
    
    grid_configs = [
        (5, 5),  # 25 cells, use 21
        (4, 6),  # 24 cells
        (6, 4),
        (3, 7),  # 21 exactly
        (7, 3),
        (4, 5),  # 20, close
        (5, 4),
        (3, 6),  # 18
        (6, 3),
    ]
    
    # Sort by expected quality: prefer near-square aspects and balanced grids
    configs = []
    for W, H in aspect_ratios:
        if W <= 0 or H <= 0:
            continue
        for nc, nr in grid_configs:
            if nc * nr >= N:
                configs.append((W, H, nc, nr))
    
    # Limit number of configs to try given time budget
    max_configs = 20
    configs = configs[:max_configs]
    
    for W, H, nc, nr in configs:
        if time.time() > deadline:
            break
        
        time_per_config = max(3.0, (deadline - time.time()) / max(1, len(configs)))
        val, circles = _optimize_for_aspect_ratio(W, H, nc, nr, time_per_config)
        
        if val > best_val:
            best_val = val
            best_circles = circles.copy()
    
    # Final fallback: if nothing valid, use simple grid
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _make_grid_initial(W, H, 5, 5)
        # Validate
        val, valid = _compute_value(best_circles)
        if not valid:
            # Very conservative fallback
            W, H = 1.0, 1.0
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) for i in range(side) for j in range(side)]
            best_circles = np.array(cells[:N])
    
    # Ensure final result is valid
    val, valid = _compute_value(best_circles)
    if not valid:
        # Last resort
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.35 / side) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles

# EVOLVE-BLOCK-END
