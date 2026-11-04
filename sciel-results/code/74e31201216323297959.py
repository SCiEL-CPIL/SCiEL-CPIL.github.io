"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 135.0  # seconds, safety margin before 150s kill


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


def _objective_and_grad(params, W, H, penalty_weight=1000.0, eps=1e-4):
    """
    Objective and gradient: minimize -sum(r) + penalties for violations.
    Vectorized for speed with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n) - 1.0  # d(-sum(r))/dr = -1

    # Smooth max helper
    def _smooth_max(v, e=eps):
        return 0.5 * (v + np.sqrt(v ** 2 + e ** 2))

    def _smooth_max_grad(v, e=eps):
        return 0.5 * (1 + v / np.sqrt(v ** 2 + e ** 2))

    # Boundary penalties
    v1 = r - x  # violation if x < r
    v2 = r - y  # violation if y < r
    v3 = x + r - W  # violation if x + r > W
    v4 = y + r - H  # violation if y + r > H
    v5 = -r  # violation if r < 0

    sm1 = _smooth_max(v1)
    sm2 = _smooth_max(v2)
    sm3 = _smooth_max(v3)
    sm4 = _smooth_max(v4)
    sm5 = _smooth_max(v5)

    obj += penalty_weight * (np.sum(sm1 ** 2) + np.sum(sm2 ** 2) +
                             np.sum(sm3 ** 2) + np.sum(sm4 ** 2) +
                             np.sum(sm5 ** 2))

    g1 = penalty_weight * 2 * sm1 * _smooth_max_grad(v1)
    g2 = penalty_weight * 2 * sm2 * _smooth_max_grad(v2)
    g3 = penalty_weight * 2 * sm3 * _smooth_max_grad(v3)
    g4 = penalty_weight * 2 * sm4 * _smooth_max_grad(v4)
    g5 = penalty_weight * 2 * sm5 * _smooth_max_grad(v5)

    # d(v1)/dx = -1, d(v1)/dr = 1
    gx -= g1
    gr += g1
    # d(v2)/dy = -1, d(v2)/dr = 1
    gy -= g2
    gr += g2
    # d(v3)/dx = 1, d(v3)/dr = 1
    gx += g3
    gr += g3
    # d(v4)/dy = 1, d(v4)/dr = 1
    gy += g4
    gr += g4
    # d(v5)/dr = -1
    gr -= g5

    # Overlap penalties: vectorized computation
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2 + 1e-24)
    np.fill_diagonal(dist_mat, 1e12)

    r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum_mat - dist_mat

    # Only use upper triangle to avoid double counting
    iu, ju = np.triu_indices(n, k=1)
    v_vec = v_overlap[iu, ju]
    pos_v = np.maximum(v_vec, 0)
    
    obj += penalty_weight * np.sum(pos_v ** 2)

    # Gradient for overlaps
    # For each pair (i,j) where i<j, if v = r_i + r_j - d_ij > 0:
    # d(obj)/d(r_i) += 2 * penalty_weight * v
    # d(obj)/d(r_j) += 2 * penalty_weight * v
    # d(obj)/d(x_i) += 2 * penalty_weight * v * (x_i - x_j) / d_ij
    # d(obj)/d(y_i) += 2 * penalty_weight * v * (y_i - y_j) / d_ij
    # d(obj)/d(x_j) -= 2 * penalty_weight * v * (x_i - x_j) / d_ij
    # d(obj)/d(y_j) -= 2 * penalty_weight * v * (y_i - y_j) / d_ij
    
    grad_factor = 2.0 * penalty_weight * pos_v
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    d_ij = dist_mat[iu, ju]
    
    # Guard against division by zero
    safe_d = np.where(d_ij > 1e-12, d_ij, 1e-12)
    
    gx[iu] += grad_factor * dx_ij / safe_d
    gy[iu] += grad_factor * dy_ij / safe_d
    gx[ju] -= grad_factor * dx_ij / safe_d
    gy[ju] -= grad_factor * dy_ij / safe_d
    gr[iu] += grad_factor
    gr[ju] += grad_factor

    grad = np.concatenate([gx, gy, gr])
    return obj, grad


def _local_optimize(W, H, initial_circles, time_budget=15.0):
    """
    Optimize the configuration using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = initial_circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = initial_circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    # Anneal penalty weight: start low, increase to enforce constraints
    weights = [5, 20, 100, 500, 2000, 10000]
    current_params = params.copy()

    for w in weights:
        if time.time() - start > time_budget - 2:
            break
        try:
            res = minimize(
                _objective_and_grad, current_params, args=(W, H, w),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            current_params = res.x
        except Exception:
            continue

        circles = current_params.reshape(N, 3)
        # Repair step
        circles, valid = _separate_circles(circles, W, H, max_iter=1000)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score


def _high_precision_polish(circles, W, H, time_budget=10.0):
    """
    High-precision polish: use very tight tolerances and more iterations.
    """
    start = time.time()
    current = circles.copy()
    best = circles.copy()
    best_score = np.sum(best[:, 2])
    
    # Use progressively tighter epsilon
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6]
    penalty_weights = [5000, 20000, 100000, 500000]
    
    params = current.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))
    
    for eps, pw in zip(epsilons, penalty_weights):
        if time.time() - start > time_budget - 1:
            break
        try:
            # Define objective with specific eps
            def obj_func(p, _W=W, _H=H, _pw=pw, _eps=eps):
                return _objective_and_grad(p, _W, _H, _pw, _eps)[0]
            
            def grad_func(p, _W=W, _H=H, _pw=pw, _eps=eps):
                return _objective_and_grad(p, _W, _H, _pw, _eps)[1]
            
            res = minimize(
                obj_func, params, jac=grad_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            params = res.x
        except Exception:
            continue
        
        circles = params.reshape(N, 3)
        circles, valid = _separate_circles(circles, W, H, max_iter=2000)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best = circles.copy()
                params = best.flatten()
    
    return best, best_score


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


def _init_hexagonal(W, H):
    """Hexagonal packing initialization for 21 circles."""
    cells = []
    # Try to fit a hexagonal pattern
    # For a rectangle, we can use rows of circles in a staggered pattern
    n_rows = 5
    n_cols = 5
    
    # Calculate optimal radius for hexagonal packing
    # In a hexagonal pattern, the distance between centers is 2r
    # Vertical spacing is sqrt(3)*r, horizontal is 2r
    # For 5 rows: total height = r + 4*sqrt(3)*r + r = 2r + 4*sqrt(3)*r
    # For 5 cols: total width = 2r + 4*2r = 10r (for even rows), odd rows offset
    
    # Let's compute r to fit in W x H
    # Height: 2r + 4*sqrt(3)*r <= H  =>  r <= H / (2 + 4*sqrt(3))
    # Width: 10r <= W  =>  r <= W/10
    
    r_h = H / (2 + 4 * np.sqrt(3))
    r_w = W / 10.0
    r = min(r_h, r_w) * 0.95  # 5% margin
    
    # Generate positions
    for i in range(n_rows):
        for j in range(n_cols):
            if len(cells) >= N:
                break
            # Stagger odd rows
            offset = (j % 2) * r if (i % 2 == 1) else 0
            cx = r + j * 2 * r + offset
            cy = r + i * np.sqrt(3) * r
            
            # Check if it fits
            if cx + r <= W and cy + r <= H:
                cells.append([cx, cy, r])
    
    # If we don't have enough, fall back to grid
    if len(cells) < N:
        return _init_grid_5x5(W, H)
    
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

    # Aspect ratios to try - expanded range
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]

    rng = np.random.default_rng(42)

    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Determine initializations based on aspect ratio
        inits = []
        if 0.85 <= ar <= 1.15:
            inits.append(_init_grid_5x5(W, H))
            inits.append(_init_hexagonal(W, H))
        elif ar < 1.0:
            inits.append(_init_grid_3x7(W, H))
            inits.append(_init_hexagonal(W, H))
        else:
            inits.append(_init_grid_3x7(W, H))

        # Always try a random initialization
        inits.append(_init_random(W, H, rng))

        for init in inits:
            if time.time() - start_time > DEADLINE - 10:
                break
            # Allocate time budget based on remaining time
            remaining = DEADLINE - 10 - (time.time() - start_time)
            if remaining < 8:
                break
            time_budget = min(12.0, remaining / 2.0)
            
            circles, score = _local_optimize(W, H, init, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = W, H

    # High-precision final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        remaining = DEADLINE - 8 - (time.time() - start_time)
        if remaining > 5:
            polished, polished_score = _high_precision_polish(best_result, best_W, best_H, time_budget=min(10.0, remaining))
            if polished_score > best_score:
                best_score = polished_score
                best_result = polished.copy()

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
