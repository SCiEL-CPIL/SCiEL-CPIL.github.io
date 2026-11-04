"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_init(n_circles, aspect):
    """Initialize circles in a hexagonal-like pattern within a rectangle of given aspect ratio."""
    W = np.sqrt(aspect)
    H = 1.0 / np.sqrt(aspect)
    
    # Estimate grid dimensions
    # For hexagonal packing, rows are offset
    # Try to find good grid layout
    best_layout = None
    best_score = -1
    
    for cols in range(1, 10):
        rows = int(np.ceil(n_circles / cols))
        if cols * rows < n_circles:
            continue
        # Check if this layout fits reasonably
        cell_w = W / cols
        cell_h = H / rows
        # Hexagonal offset
        if rows > 1:
            # Effective height per row is cell_h * sqrt(3)/2 for hex
            pass
        score = min(cell_w, cell_h)
        if score > best_score:
            best_score = score
            best_layout = (cols, rows)
    
    if best_layout is None:
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
    else:
        cols, rows = best_layout
    
    circles = []
    for i in range(cols):
        for j in range(rows):
            if len(circles) >= n_circles:
                break
            # Hexagonal offset: odd rows shifted by half cell width
            x_offset = 0.5 / cols if (j % 2 == 1) else 0
            cx = (i + 0.5 + x_offset) / cols * W
            cy = (j + 0.5) / rows * H
            circles.append([cx, cy, 0.01])
    
    # Fill remaining if needed
    while len(circles) < n_circles:
        circles.append([0.5 * W, 0.5 * H, 0.01])
    
    return np.array(circles[:n_circles])


def _objective_and_grad(x, n_circles, W, H, mu):
    """
    Objective: minimize -sum(r) + mu * penalty
    where penalty = sum(max(0, r_i + r_j - dist_ij)^2) + sum(max(0, r_i - min(cx, W-cx, cy, H-cy))^2)
    
    x = [cx1, cy1, r1, cx2, cy2, r2, ...]
    """
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Compute pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Overlap penalty: for each pair, max(0, r_i + r_j - dist)^2
    overlap_sum = rs[:, None] + rs[None, :] - dist
    overlap_pos = np.maximum(overlap_sum, 0)
    # Exclude diagonal
    np.fill_diagonal(overlap_pos, 0)
    overlap_penalty = np.sum(overlap_pos**2)
    
    # Boundary penalty: circle must be inside [0, W] x [0, H]
    # For each circle, min distance to boundary
    min_dist = np.minimum(np.minimum(xs, W - xs), np.minimum(ys, H - ys))
    boundary_excess = np.maximum(rs - min_dist, 0)
    boundary_penalty = np.sum(boundary_excess**2)
    
    # Also ensure positive radii
    neg_r = np.maximum(-rs, 0)
    neg_penalty = np.sum(neg_r**2)
    
    total_penalty = overlap_penalty + boundary_penalty + neg_penalty
    
    obj = -np.sum(rs) + mu * total_penalty
    
    # Gradient
    grad = np.zeros_like(x)
    
    # d(-sum(rs))/dx = -1 for r components
    grad[2::3] -= 1.0
    
    # Gradient of overlap penalty
    # d(overlap_pos_ij^2)/dx_k for k in {i, j}
    # For pair (i,j): if overlap_pos_ij > 0, gradient contribution
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            op = overlap_pos[i, j]
            if op > 0:
                d_dist_dx = (xs[i] - xs[j]) / dist[i, j] if dist[i, j] > 1e-10 else 0
                d_dist_dy = (ys[i] - ys[j]) / dist[i, j] if dist[i, j] > 1e-10 else 0
                # d(op^2)/dx_i = 2*op * (-d_dist/dx_i) = 2*op * d_dist_dx
                # d(op)/dx_i = -d(dist)/dx_i = -(xs[i]-xs[j])/dist = -d_dist_dx
                # Wait: op = r_i + r_j - dist
                # d(op)/dx_i = -d(dist)/dx_i = -(xs[i]-xs[j])/dist
                # d(op^2)/dx_i = 2*op * d(op)/dx_i
                grad[3*i + 0] += mu * 2 * op * (-(xs[i] - xs[j]) / max(dist[i, j], 1e-10))
                grad[3*j + 0] += mu * 2 * op * (-(xs[j] - xs[i]) / max(dist[i, j], 1e-10))
                grad[3*i + 1] += mu * 2 * op * (-(ys[i] - ys[j]) / max(dist[i, j], 1e-10))
                grad[3*j + 1] += mu * 2 * op * (-(ys[j] - ys[i]) / max(dist[i, j], 1e-10))
                grad[3*i + 2] += mu * 2 * op
                grad[3*j + 2] += mu * 2 * op
    
    # Gradient of boundary penalty
    for i in range(n_circles):
        be = boundary_excess[i]
        if be > 0:
            # min_dist = min(xs, W-xs, ys, H-ys)
            # We need to know which constraint is active
            d_left = xs[i]
            d_right = W - xs[i]
            d_bottom = ys[i]
            d_top = H - ys[i]
            min_d = min(d_left, d_right, d_bottom, d_top)
            # d(min_dist)/dx_i = -1 if left is active, +1 if right is active, 0 otherwise
            # d(min_dist)/dy_i = -1 if bottom is active, +1 if top is active, 0 otherwise
            # d(be)/dx_i = -d(min_dist)/dx_i (since be = max(rs - min_dist, 0))
            # d(be^2)/dx_i = 2*be * d(be)/dx_i = -2*be * d(min_dist)/dx_i
            
            if abs(d_left - min_d) < 1e-12:
                grad[3*i + 0] += mu * 2 * be * (-1)
            elif abs(d_right - min_d) < 1e-12:
                grad[3*i + 0] += mu * 2 * be * (1)
            
            if abs(d_bottom - min_d) < 1e-12:
                grad[3*i + 1] += mu * 2 * be * (-1)
            elif abs(d_top - min_d) < 1e-12:
                grad[3*i + 1] += mu * 2 * be * (1)
            
            grad[3*i + 2] += mu * 2 * be
    
    # Gradient of neg_penalty
    for i in range(n_circles):
        if rs[i] < 0:
            grad[3*i + 2] += mu * 2 * (-rs[i])
    
    return obj, grad


def _check_validity(circles, W, H, tol=1e-6):
    """Check if circles are valid (disjoint, inside rectangle)."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r < -tol:
            return False
        if cx - r < -tol or cx + r > W + tol or cy - r < -tol or cy + r > H + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_sum_r = -1
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.35]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        W = np.sqrt(aspect)
        H = 1.0 / np.sqrt(aspect)
        
        # Hexagonal initialization
        x0 = _hex_init(N, aspect)
        
        # Start with moderate mu, increase if needed
        mu = 10.0
        x_current = x0.copy()
        
        # Iteratively optimize with increasing mu
        for mu_iter in range(8):
            if time.time() > deadline - 3.0:
                break
            
            try:
                res = minimize(
                    _objective_and_grad,
                    x_current,
                    args=(N, W, H, mu),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
                )
                x_current = res.x
            except Exception:
                break
            
            # Check penalty
            xs = x_current[0::3]
            ys = x_current[1::3]
            rs = x_current[2::3]
            
            # Check overlap
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            overlap_sum = rs[:, None] + rs[None, :] - dist
            np.fill_diagonal(overlap_sum, 0)
            max_overlap = np.max(overlap_sum)
            
            # Check boundary
            min_dist = np.minimum(np.minimum(xs, W - xs), np.minimum(ys, H - ys))
            max_boundary_excess = np.max(rs - min_dist)
            
            max_violation = max(max_overlap, max_boundary_excess, np.max(-rs) if len(rs) > 0 else 0)
            
            if max_violation < 1e-4:
                break
            else:
                mu *= 5.0
        
        # Final polish with high mu
        if time.time() < deadline - 2.0:
            try:
                res = minimize(
                    _objective_and_grad,
                    x_current,
                    args=(N, W, H, 1000.0),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                x_current = res.x
            except Exception:
                pass
        
        # Extract circles and validate
        xs = x_current[0::3]
        ys = x_current[1::3]
        rs = np.maximum(x_current[2::3], 0)
        
        circles = np.column_stack([xs, ys, rs])
        
        # Clip to ensure inside rectangle
        for i in range(len(circles)):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
        
        if _check_validity(circles, W, H):
            sum_r = np.sum(circles[:, 2])
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = circles.copy()
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
