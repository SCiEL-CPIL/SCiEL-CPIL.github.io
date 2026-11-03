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
    Uses smooth penalty formulation.
    params: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{N-1}, y_{N-1}, r_{N-1}]
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Penalty for negative radii (smooth)
    pen = 0.0
    neg_r = np.minimum(rs, 0)
    if np.any(neg_r < 0):
        pen += 100.0 * np.sum(neg_r**2)
    
    # Boundary penalties (smooth)
    bx_pen = 0.0
    for i in range(n):
        bx_pen += max(0, -(xs[i] - rs[i]))**2
        bx_pen += max(0, (xs[i] + rs[i]) - W)**2
        bx_pen += max(0, -(ys[i] - rs[i]))**2
        bx_pen += max(0, (ys[i] + rs[i]) - H)**2
    
    # Overlap penalties (smooth)
    ov_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist**2:
                dist = np.sqrt(dist_sq)
                overlap = min_dist - dist
                ov_pen += overlap**2
    
    total_penalty = 100.0 * (bx_pen + ov_pen)
    
    return -np.sum(rs) + total_penalty


def _make_grid_initial(W, H, n_cols, n_rows):
    """Create initial grid of circles, ensuring exactly N circles."""
    circles = []
    total_cells = n_cols * n_rows
    for i in range(n_cols):
        for j in range(n_rows):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / n_cols
            cy = (j + 0.5) * H / n_rows
            # Use a slightly smaller radius to ensure non-overlap initially
            cell_w = W / n_cols
            cell_h = H / n_rows
            r = min(cell_w, cell_h) * 0.45
            circles.append([cx, cy, r])
    # If we have fewer than N, add more at random positions
    while len(circles) < N:
        cx = W * 0.5
        cy = H * 0.5
        r = min(W, H) * 0.05
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
    
    # Try multiple random restarts with Nelder-Mead
    n_restarts = 4
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit * 0.6:
            break
        
        if restart == 0:
            p = params.copy()
        else:
            p = params.copy()
            # Perturb
            p[0::3] += rng.normal(0, W*0.05, N)
            p[1::3] += rng.normal(0, H*0.05, N)
            p[2::3] *= (1 + rng.normal(0, 0.15, N))
            p[2::3] = np.clip(p[2::3], 0.01, min(W, H)/2)
        
        res = minimize(_objective, p, args=(W, H), method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    # Final polish pass with L-BFGS-B for higher precision
    if time.time() - start < time_limit * 0.8:
        p = best_params.copy()
        # Define bounds for L-BFGS-B
        bounds = []
        for i in range(N):
            r_max = min(W, H) / 2
            bounds.append((1e-6, W))      # x
            bounds.append((1e-6, H))      # y
            bounds.append((1e-6, r_max))  # r
        
        res2 = minimize(_objective, p, args=(W, H), method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_params = res2.x
    
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
        (1.2, 0.8),       # slightly wide
        (0.8, 1.2),       # slightly tall
        (1.5, 0.5),       # wide
        (0.5, 1.5),       # tall
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.4, 0.6),
        (0.6, 1.4),
    ]
    
    grid_configs = [
        (5, 5),  # 25 cells, use 21
        (3, 7),  # 21 exactly
        (7, 3),
        (4, 6),  # 24 cells
        (6, 4),
        (5, 4),  # 20, add 1
        (4, 5),
        (3, 8),  # 24
        (8, 3),
    ]
    
    # Build configs, prioritizing near-square aspects and balanced grids
    configs = []
    for W, H in aspect_ratios:
        if W <= 0 or H <= 0:
            continue
        for nc, nr in grid_configs:
            if nc * nr >= N:
                configs.append((W, H, nc, nr))
    
    # Sort configs: prefer square aspect ratio and balanced grid
    def config_score(c):
        W, H, nc, nr = c
        aspect_penalty = abs(W/H - 1.0)
        grid_penalty = abs(nc - nr)
        return aspect_penalty + 0.1 * grid_penalty
    
    configs.sort(key=config_score)
    
    # Limit number of configs to try given time budget
    max_configs = 15
    configs = configs[:max_configs]
    
    for idx, (W, H, nc, nr) in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Allocate more time to early configs, less to later ones
        remaining_configs = len(configs) - idx
        time_per_config = max(2.0, (deadline - time.time()) / max(1, remaining_configs))
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
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.38 / side) for i in range(side) for j in range(side)]
            best_circles = np.array(cells[:N])
    
    # Ensure final result is valid
    val, valid = _compute_value(best_circles)
    if not valid:
        # Last resort: even more conservative
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.35 / side) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles

# EVOLVE-BLOCK-END
