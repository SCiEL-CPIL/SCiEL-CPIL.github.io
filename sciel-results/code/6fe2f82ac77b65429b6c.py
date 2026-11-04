"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Structural change: SLSQP-based constrained optimization on true objective + collision resolution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_solution(circles, W, H, tol=1e-7):
    """Validate that circles are within bounds and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x < r - tol or x > W - r + tol:
            return False
        if y < r - tol or y > H - r + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True


def _resolve_collisions(circles, W, H, iterations=200):
    """Resolve overlaps by pushing circles apart. Returns modified circles."""
    n = len(circles)
    circles = circles.copy()
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = circles[i, 2] + circles[j, 2]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    push = overlap / 2.0 + 1e-6
                    nx, ny = dx / d, dy / d
                    circles[i, 0] += nx * push
                    circles[i, 1] += ny * push
                    circles[j, 0] -= nx * push
                    circles[j, 1] -= ny * push
                    moved = True
                elif d <= 1e-12:
                    # Random nudge
                    ang = np.random.uniform(0, 2 * np.pi)
                    push = (min_d / 2.0 + 1e-6)
                    circles[i, 0] += np.cos(ang) * push
                    circles[i, 1] += np.sin(ang) * push
                    circles[j, 0] -= np.cos(ang) * push
                    circles[j, 1] -= np.sin(ang) * push
                    moved = True
        # Clamp to bounds
        for i in range(n):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, W - r)
            circles[i, 1] = np.clip(circles[i, 1], r, H - r)
        if not moved:
            break
    return circles


def _generate_grid_init(W, H, n=N):
    """Generate a grid initialization for n circles in W x H."""
    cols = max(1, int(np.round(np.sqrt(n * W / H))))
    rows = max(1, int(np.ceil(n / cols)))
    positions = np.zeros((n, 2))
    for i in range(n):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * W / cols
        positions[i, 1] = (row + 0.5) * H / rows
    return positions


def _generate_hex_init(W, H, n=N):
    """Generate a hexagonal lattice initialization."""
    positions = []
    spacing = 1.0
    row_spacing = spacing * np.sqrt(3) / 2
    row = 0
    while len(positions) < n:
        remaining = n - len(positions)
        for col in range(remaining):
            x = col * spacing + (row % 2) * spacing / 2
            y = row * row_spacing
            positions.append([x, y])
            if len(positions) >= n:
                break
        row += 1
    positions = np.array(positions[:n])
    # Normalize to fit in W x H
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    bw, bh = max_x - min_x, max_y - min_y
    if bw > 0 and bh > 0:
        scale = min(W / (bw + 0.3), H / (bh + 0.3))
        positions = positions * scale
        positions[:, 0] += (W - positions[:, 0].max() + positions[:, 0].min()) / 2
        positions[:, 1] += (H - positions[:, 1].max() + positions[:, 1].min()) / 2
    return positions


def _solve_aspect(W, H, deadline, init_method='grid'):
    """Solve packing for a given aspect ratio using SLSQP optimization."""
    start_time = time.time()
    
    # Choose initialization
    if init_method == 'hex':
        positions = _generate_hex_init(W, H)
    else:
        positions = _generate_grid_init(W, H)
    
    # Initial uniform radius
    def _max_uniform_radius(pos):
        n = len(pos)
        min_x, min_y = pos.min(axis=0)
        max_x, max_y = pos.max(axis=0)
        r_bound = min(min_x, W - max_x, min_y, H - max_y)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = dists.min() / 2.0
        return max(0.0, min(r_bound, r_pair))
    
    r_init = _max_uniform_radius(positions)
    
    # Objective: minimize negative of minimum clearance (i.e., maximize min clearance)
    # Variables: 2*n positions + 1 radius = 2*n+1 variables
    # We optimize positions and radius jointly
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        if r < 0:
            return 1e10
        
        # Min clearance: min of (pairwise dist - 2r) and (distance to walls - r)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() - 2 * r
        
        wall_clear = np.minimum(
            np.minimum(pos[:, 0], W - pos[:, 0]),
            np.minimum(pos[:, 1], H - pos[:, 1])
        )
        min_wall = wall_clear.min() - r
        
        min_clear = min(min_pair, min_wall)
        return -min_clear
    
    def objective_grad(x):
        pos = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Find which constraint is active
        min_pair_idx = np.argmin(dists)
        i, j = divmod(min_pair_idx, N)
        if i == j:
            i, j = 0, 1
        
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        
        wall_clear = np.minimum(
            np.minimum(pos[:, 0], W - pos[:, 0]),
            np.minimum(pos[:, 1], H - pos[:, 1])
        )
        min_wall_idx = np.argmin(wall_clear)
        
        min_pair_val = dists[i, j] - 2 * r
        min_wall_val = wall_clear[min_wall_idx] - r
        
        grad = np.zeros(2 * N + 1)
        
        if min_pair_val <= min_wall_val:
            # Pair constraint active
            grad[2 * i] = dx / d
            grad[2 * i + 1] = dy / d
            grad[2 * j] = -dx / d
            grad[2 * j + 1] = -dy / d
            grad[2 * N] = 2.0
        else:
            # Wall constraint active
            k = min_wall_idx
            xk, yk = pos[k]
            if xk <= W - xk and xk <= yk and xk <= H - yk:
                grad[2 * k] = 1.0
            elif W - xk <= xk and W - xk <= yk and W - xk <= H - yk:
                grad[2 * k] = -1.0
            elif yk <= xk and yk <= W - xk and yk <= H - yk:
                grad[2 * k + 1] = 1.0
            else:
                grad[2 * k + 1] = -1.0
            grad[2 * N] = 1.0
        
        return -grad
    
    # Constraints: positions within bounds, radius >= 0
    # We use bounds on SLSQP
    
    x0 = np.zeros(2 * N + 1)
    x0[:2 * N] = positions.flatten()
    x0[2 * N] = r_init
    
    bounds = []
    for i in range(N):
        bounds.append((r_init * 0.5, W - r_init * 0.5))
        bounds.append((r_init * 0.5, H - r_init * 0.5))
    bounds.append((0.0, min(W, H) / 2.0))
    
    best_result = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    inits = []
    inits.append(('grid', _generate_grid_init(W, H)))
    inits.append(('hex', _generate_hex_init(W, H)))
    # Add perturbed grid
    pert = _generate_grid_init(W, H) + np.random.randn(N, 2) * 0.02 * W
    pert[:, 0] = np.clip(pert[:, 0], 0.05 * W, 0.95 * W)
    pert[:, 1] = np.clip(pert[:, 1], 0.05 * H, 0.95 * H)
    inits.append(('perturbed', pert))
    
    for init_name, pos in inits:
        if time.time() > deadline:
            break
        
        r_init = _max_uniform_radius(pos)
        if r_init <= 0:
            continue
        
        x0 = np.zeros(2 * N + 1)
        x0[:2 * N] = pos.flatten()
        x0[2 * N] = r_init
        
        bounds = []
        for i in range(N):
            bounds.append((r_init * 0.3, W - r_init * 0.3))
            bounds.append((r_init * 0.3, H - r_init * 0.3))
        bounds.append((r_init * 0.5, min(W, H) / 2.0))
        
        try:
            result = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if time.time() < deadline:
                pos_opt = result.x[:2 * N].reshape(N, 2)
                r_opt = result.x[2 * N]
                
                # Clamp
                pos_opt[:, 0] = np.clip(pos_opt[:, 0], r_opt, W - r_opt)
                pos_opt[:, 1] = np.clip(pos_opt[:, 1], r_opt, H - r_opt)
                
                circles = np.column_stack([pos_opt, np.full(N, r_opt)])
                
                # Collision resolution
                circles = _resolve_collisions(circles, W, H, iterations=100)
                
                # Recompute actual max radius after resolution
                r_actual = _max_uniform_radius(circles[:, :2])
                circles[:, 2] = r_actual
                
                if _validate_solution(circles, W, H):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = circles.copy()
                        
        except Exception:
            pass
    
    # If SLSQP failed, try collision resolution on initial grid
    if best_result is None:
        pos = _generate_grid_init(W, H)
        r = _max_uniform_radius(pos)
        if r > 0:
            circles = np.column_stack([pos, np.full(N, r)])
            circles = _resolve_collisions(circles, W, H, iterations=300)
            r_actual = _max_uniform_radius(circles[:, :2])
            circles[:, 2] = r_actual
            if _validate_solution(circles, W, H):
                best_result = circles
                best_score = np.sum(circles[:, 2])
    
    return best_result, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.83, 1.33, 0.75, 1.5, 0.67, 1.4, 0.7, 1.25, 0.8, 1.6, 0.6]
    
    # Sort by proximity to square (likely optimal for circle packing)
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.0))
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Try both grid and hex initializations
        for init_method in ['grid', 'hex']:
            if time.time() > deadline:
                break
            
            result, score = _solve_aspect(w, h, deadline, init_method)
            if result is not None and score > best_score:
                best_score = score
                best_result = result.copy()
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in unit square
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
