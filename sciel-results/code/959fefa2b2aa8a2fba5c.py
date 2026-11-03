"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=2000):
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

        # Resolve boundary violations by shrinking radii if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            # Clamp position to keep circle inside rectangle
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    # Push apart
                    if dist < 1e-12:
                        # Random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    # Distribute push based on relative sizes
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            # If circle can't fit at current position, shrink it
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


def _objective_and_grad(params, W, H, penalty_weight):
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
    eps = 1e-4

    def _smooth_max_grad(v):
        """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = -(x - r) = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += penalty_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * penalty_weight * smax * gsmax
    grad[2::3] += 2.0 * penalty_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += penalty_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * penalty_weight * smax * gsmax
    grad[2::3] += 2.0 * penalty_weight * smax * gsmax

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += penalty_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * penalty_weight * smax * gsmax
    grad[2::3] += 2.0 * penalty_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += penalty_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * penalty_weight * smax * gsmax
    grad[2::3] += 2.0 * penalty_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += penalty_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * penalty_weight * smax * gsmax

    # Overlap penalties: vectorized computation
    # Compute all pairwise distances
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    # Avoid division by zero
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
    
    # Only use upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]
    
    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += penalty_weight * np.sum(smax_vec ** 2)
    
    factor_vec = 2.0 * penalty_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij
    
    # Accumulate gradients using np.add.at for repeated indices
    grad_x_i = factor_vec * nx_ij
    grad_y_i = factor_vec * ny_ij
    grad_x_j = -factor_vec * nx_ij
    grad_y_j = -factor_vec * ny_ij
    
    np.add.at(grad, 3 * iu, grad_x_i)
    np.add.at(grad, 3 * iu + 1, grad_y_i)
    np.add.at(grad, 3 * iu + 2, factor_vec)
    np.add.at(grad, 3 * ju, grad_x_j)
    np.add.at(grad, 3 * ju + 1, grad_y_j)
    np.add.at(grad, 3 * ju + 2, factor_vec)

    return obj, grad


def _local_optimize(circles, W, H, time_budget=10.0):
    """
    Optimize the configuration using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    # Anneal penalty weight: start low, increase to enforce constraints
    weights = [10, 50, 200, 1000, 5000, 20000]
    current_params = params.copy()

    for w in weights:
        if time.time() - start > time_budget - 2:
            break
        try:
            res = minimize(
                _objective_and_grad, current_params, args=(W, H, w),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            current_params = res.x
        except Exception:
            continue

        circles_opt = current_params.reshape(N, 3)
        # Repair step
        circles_rep, valid = _separate_circles(circles_opt, W, H, max_iter=500)
        if valid:
            score = np.sum(circles_rep[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles_rep.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score


def _init_hexagonal(W, H, rows, cols):
    """
    Initialize a hexagonal packing.
    Returns array of shape (n, 3) with n = rows * cols (if <= N) or N.
    """
    # Hexagonal packing:
    # Row 0: x = r, 3r, 5r, ...
    # Row 1: x = 2r, 4r, 6r, ... (shifted by r)
    # Vertical spacing: sqrt(3) * r
    
    # We want to fit rows * cols circles.
    # Let's determine radius r such that it fits in W x H.
    # Width required: (cols - 1) * 2r + 2r = cols * 2r ? No.
    # In hex packing, horizontal distance is 2r.
    # Width = (cols - 1) * 2r + 2r = cols * 2r.
    # Height = (rows - 1) * sqrt(3) * r + 2r.
    
    # We need to choose r such that cols * 2r <= W and (rows - 1) * sqrt(3) * r + 2r <= H.
    # r <= W / (2 * cols)
    # r <= H / (2 + (rows - 1) * sqrt(3))
    
    r1 = W / (2.0 * cols)
    r2 = H / (2.0 + (rows - 1) * np.sqrt(3))
    r = min(r1, r2) * 0.999  # Small margin
    
    circles = []
    for i in range(rows):
        for j in range(cols):
            # x position
            # Offset row i by (i % 2) * r
            x = r + j * 2 * r + (i % 2) * r
            # y position
            y = r + i * np.sqrt(3) * r
            
            # Ensure inside bounds (center must be at least r away from edge)
            # The above formulas ensure x >= r and y >= r.
            # Check upper bounds:
            # x_max = r + (cols-1)*2r + r = cols*2r. If cols*2r <= W, then x_max <= W.
            # But we need x + r <= W => x <= W - r.
            # If W = cols * 2r, then x_max = W. x_max + r = W + r > W. Violation.
            # So we need to scale down or adjust.
            # My r calculation: r = W / (2 * cols).
            # Max x = r + (cols-1)*2r + r = 2r*cols = W.
            # So x_max = W. But we need x <= W - r.
            # So we need W - r >= W => r <= 0. Impossible.
            # The width of the bounding box of the centers is (cols-1)*2r + r (offset) + r (offset) ?
            # Min x = r. Max x = r + (cols-1)*2r + r = 2r*cols.
            # Width of centers span = 2r*cols - r = r(2cols - 1).
            # Plus radius on both sides: r(2cols - 1) + 2r = r(2cols + 1).
            # So we need r(2cols + 1) <= W => r <= W / (2cols + 1).
            
            # Let's re-calculate r properly.
            pass
            
    # Recalculate r with correct bounds
    # Horizontal: r(2*cols + 1) <= W  => r <= W / (2*cols + 1)
    # Vertical:   r(2 + (rows-1)*sqrt(3)) <= H => r <= H / (2 + (rows-1)*sqrt(3))
    
    r_h = W / (2.0 * cols + 1.0)
    r_v = H / (2.0 + (rows - 1) * np.sqrt(3))
    r = min(r_h, r_v) * 0.999
    
    circles = []
    for i in range(rows):
        for j in range(cols):
            x = r + j * 2 * r + (i % 2) * r
            y = r + i * np.sqrt(3) * r
            
            # Clamp to be safe
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            circles.append([x, y, r])
            
    return np.array(circles[:N])


def _init_grid_5x5(W, H):
    """5x5 grid minus 4 corners (21 circles)."""
    cells = []
    side = 5
    for i in range(side):
        for j in range(side):
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _init_grid_3x7(W, H):
    """3x7 grid (21 circles)."""
    cells = []
    rows = 3
    cols = 7
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _init_random(W, H, rng):
    """Random initialization."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.05, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios to try
    # Broader range as per inspiration
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Determine initializations
        inits = []
        
        # Hexagonal lattice initializations (new strategy)
        # Try different row/col splits for 21 circles
        # 3x7, 7x3
        if 0.8 <= ar <= 1.2:
            # Square-ish
            inits.append(_init_hexagonal(W, H, 3, 7))
            inits.append(_init_hexagonal(W, H, 7, 3))
            inits.append(_init_grid_5x5(W, H))
        else:
            # Elongated
            inits.append(_init_hexagonal(W, H, 3, 7))
            inits.append(_init_hexagonal(W, H, 7, 3))
            inits.append(_init_grid_3x7(W, H))

        # Always try a random initialization
        inits.append(_init_random(W, H, rng))

        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            remaining = DEADLINE - 5 - (time.time() - start_time)
            if remaining < 5:
                break
            time_budget = min(12.0, remaining / 3.0)  # More conservative budget per init
            
            circles, score = _local_optimize(W, H, init, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = W, H

    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 3:
        # High precision polish
        polished, valid = _separate_circles(best_result, best_W, best_H, max_iter=5000)
        if valid:
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_result = polished.copy()
        
        # Try one more optimization pass on the polished result if time permits
        if time.time() - start_time < DEADLINE - 10:
            circles, score = _local_optimize(best_result, best_W, best_H, time_budget=DEADLINE - time.time() - 10)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = best_W, best_H # W,H don't change

    # Fallback
    if best_result is None:
        best_result = _init_grid_5x5(1.0, 1.0)
        best_result, _ = _separate_circles(best_result, 1.0, 1.0)
        best_W, best_H = 1.0, 1.0

    if not _check_validity(best_result, best_W, best_H):
        # Emergency fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END
