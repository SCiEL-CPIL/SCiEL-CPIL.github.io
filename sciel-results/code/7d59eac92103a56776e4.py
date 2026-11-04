"""
Strategy: Penalty-based L-BFGS-B optimization for circle packing in a variable rectangle.
Combines the structural approach of the SCR program (aspect ratio search, validation)
with the efficient penalty-based smooth objective and L-BFGS-B solver from the inspiration.
Adds hexagonal grid initialization and a final high-precision polish pass.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=500, tol=1e-10):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Clamp position to keep circle inside rectangle
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, gsmax


def _objective_and_grad(params, W, H, boundary_weight=200.0, overlap_weight=200.0, eps=1e-4):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/dr

    # Boundary penalties (smooth)
    v = r - x
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = r - y
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = x + r - W
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = y + r - H
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    v = -r
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v, eps)
            obj += overlap_weight * smax ** 2

            factor = 2.0 * overlap_weight * smax * gsmax
            d_obj_dx_i = factor * (-dx / dist)
            d_obj_dy_i = factor * (-dy / dist)
            d_obj_dx_j = factor * (dx / dist)
            d_obj_dy_j = factor * (dy / dist)
            d_obj_dr_i = factor * 1.0
            d_obj_dr_j = factor * 1.0

            grad[3 * i] += d_obj_dx_i
            grad[3 * i + 1] += d_obj_dy_i
            grad[3 * i + 2] += d_obj_dr_i
            grad[3 * j] += d_obj_dx_j
            grad[3 * j + 1] += d_obj_dy_j
            grad[3 * j + 2] += d_obj_dr_j

    return obj, grad


def _local_optimize(circles, W, H, max_iter=2000, xatol=1e-10, fatol=1e-12,
                    boundary_weight=200.0, overlap_weight=200.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight, overlap_weight)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-10, 'maxls': 100}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=5000):
    """High-precision polish pass with L-BFGS-B and tight tolerances."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight=500.0, overlap_weight=500.0, eps=1e-5)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create a hexagonal grid-based initial configuration."""
    # Try to fit a hexagonal packing pattern
    # For 21 circles, we can try different arrangements
    # A 5x5 grid with hexagonal offset
    
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Try a hexagonal arrangement
    # Rows: 0-4, with alternating offsets
    row_heights = []
    cols_per_row = []
    
    # Try to fit as many rows as possible
    # Start with a base radius and adjust
    base_r = min(W, H) / 10.0
    r = base_r * 0.8
    
    # Hexagonal spacing
    dx = 2 * r * 0.95  # slightly less than 2r to allow for optimization
    dy = r * np.sqrt(3) * 0.95
    
    row = 0
    col = 0
    idx = 0
    
    while idx < N:
        y_pos = (row + 0.5) * dy + r
        if y_pos + r > H:
            break
        
        # Determine how many circles fit in this row
        offset = 0.0
        if row % 2 == 1:
            offset = dx / 2
        
        x_pos = offset + r
        while x_pos + r <= W and idx < N:
            pos[idx, 0] = x_pos
            pos[idx, 1] = y_pos
            radii[idx] = r
            idx += 1
            x_pos += dx
        
        row += 1
    
    # If we didn't fill all circles, place remaining in a grid
    if idx < N:
        # Fallback to simple grid for remaining
        side = int(np.ceil(np.sqrt(N)))
        for i in range(idx, N):
            row_i = i // side
            col_i = i % side
            pos[i, 0] = (col_i + 0.5) * (W / side)
            pos[i, 1] = (row_i + 0.5) * (H / side)
            radii[i] = 0.5 * min(W, H) / side * 0.5
    
    # Scale all radii to ensure they fit
    min_dim = min(W, H)
    max_possible_r = min_dim / 6.0
    scale_factor = min(1.0, max_possible_r / np.max(radii)) if np.max(radii) > 0 else 1.0
    radii *= scale_factor
    
    # Adjust positions to center circles
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], radii[i], W - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], H - radii[i])
    
    return np.column_stack([pos, radii])


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) * (W / side)
        pos[i, 1] = (row + 0.5) * (H / side)
        radii[i] = 0.5 * min(W, H) / side * 0.4
    
    # Scale radii to fit
    min_dim = min(W, H)
    max_possible_r = min_dim / (side + 1)
    scale_factor = min(1.0, max_possible_r / np.max(radii)) if np.max(radii) > 0 else 1.0
    radii *= scale_factor
    
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], radii[i], W - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], H - radii[i])
    
    return np.column_stack([pos, radii])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25, 1.1, 0.9, 1.4, 0.7]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Try both hexagonal and grid initialization
        for init_type in ['hex', 'grid']:
            if time.time() > deadline - 5.0:
                break
                
            if init_type == 'hex':
                circles = _hexagonal_initialization(w, h)
            else:
                circles = _grid_initialization(w, h)
            
            # Initial separation to resolve any overlaps
            circles, _ = _separate_circles(circles, w, h, max_iter=200)
            
            # Main optimization
            circles = _local_optimize(circles, w, h, max_iter=1500)
            
            # Check validity and polish if valid
            if _check_validity(circles, w, h):
                # Polish pass
                if time.time() < deadline - 3.0:
                    circles = _polish(circles, w, h, max_iter=3000)
                    # Re-check validity after polish
                    if not _check_validity(circles, w, h):
                        # If polish made it invalid, try to repair
                        circles, valid = _separate_circles(circles, w, h, max_iter=100)
                        if valid:
                            # Final quick polish
                            if time.time() < deadline - 1.0:
                                circles = _polish(circles, w, h, max_iter=1000)
                        else:
                            # Fall back to pre-polish version
                            pass
            
            # Final validation
            if _check_validity(circles, w, h):
                value = np.sum(circles[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = circles.copy()
            else:
                # Attempt repair
                circles_repaired, valid = _separate_circles(circles, w, h, max_iter=200)
                if valid:
                    value = np.sum(circles_repaired[:, 2])
                    if value > best_value:
                        best_value = value
                        best_result = circles_repaired.copy()

    if best_result is None:
        # Fallback: simple valid grid
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])
        
    return best_result
