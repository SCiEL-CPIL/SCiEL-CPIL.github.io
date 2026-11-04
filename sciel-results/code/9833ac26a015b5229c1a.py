"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 10.0

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=500, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 0:
                    # Push apart
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Uses log-sum-exp based smooth max for better gradient behavior.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    def _soft_max(v, scale):
        """Soft max of v and 0 using log-sum-exp: scale * log(1 + exp(v/scale))."""
        v_safe = np.clip(v / scale, -50, 50)
        smax = scale * np.log1p(np.exp(v_safe))
        sig = 1.0 / (1.0 + np.exp(-v_safe))
        return smax, sig

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _soft_max(v_vec, eps)
    obj += overlap_weight * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

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

def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize. Two-stage: coarse then fine."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Stage 1: coarser eps for faster convergence
    result = minimize(
        _objective_and_grad, params, args=(W, H, 5e-4, 150.0, 150.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200}
    )
    params = result.x

    # Stage 2: tighter eps for precision
    result = minimize(
        _objective_and_grad, params, args=(W, H, 1e-5, 400.0, 400.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
    )
    return result.x.reshape(N, 3)

def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps and increasing penalty."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive refinement: 6 passes with decreasing eps and increasing weight
    eps_schedule = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    weight_schedule = [200.0, 300.0, 400.0, 500.0, 600.0, 800.0]

    for eps, wt in zip(eps_schedule, weight_schedule):
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps, wt, wt),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // len(eps_schedule), 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
        )
        params = result.x

    return params.reshape(N, 3)

def _make_hexagonal_grid(w, h, rows=None):
    """Create initial hexagonal-like arrangement for 21 circles."""
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            # Offset alternating rows for hexagonal packing
            avg_w = w / 5.0 
            offset = 0.5 * avg_w if row_i % 2 == 1 else 0
            # Distribute columns evenly across width, with offset
            x = (col_j + 0.5) * w / 5.0 + offset
            # Initial radius estimate
            r = 0.35 * min(w/5.0, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            if stagger and j % 2 == 1:
                x = (i + 0.5) * w / nc + 0.3 * w / nc
            else:
                x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    
    pos = np.array(positions[:N])
    r = 0.35 * min(w/nc, h/nr)
    rad = np.full(N, r)
    return np.column_stack((pos, rad))

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_w = None
    best_h = None

    # Strategy 1: Fixed aspect ratios with hexagonal and grid initializations
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try hexagonal init
        if time.time() < deadline - 5.0:
            circles = _make_hexagonal_grid(w, h)
            circles = _relax_circles(circles, w, h, iterations=100)
            
            # Use the improved local optimizer
            result = _local_optimize(circles, w, h, max_iter=1000)
            if _validate(result, w, h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result
                    best_w = w
                    best_h = h

        # Try grid inits
        layouts = [
            (5, 4, True),
            (4, 5, True),
            (5, 4, False),
            (4, 5, False),
        ]
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            circles = _make_grid(w, h, nc, nr, stagger)
            circles = _relax_circles(circles, w, h, iterations=100)
            
            result = _local_optimize(circles, w, h, max_iter=1000)
            if _validate(result, w, h):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result
                    best_w = w
                    best_h = h

    # Strategy 2: Variable aspect ratio optimization
    if time.time() < deadline - 30.0:
        for w_init in [1.0, 0.8, 1.2, 0.6, 1.4, 0.9, 1.1]:
            if time.time() > deadline - 15.0:
                break
            
            h_init = 2.0 - w_init
            circles = _make_hexagonal_grid(w_init, h_init)
            circles = _relax_circles(circles, w_init, h_init, iterations=100)
            
            # Optimize with fixed aspect ratio first, then adjust
            result = _local_optimize(circles, w_init, h_init, max_iter=500)
            if _validate(result, w_init, h_init):
                value = _total_radius(result)
                if value > best_value:
                    best_value = value
                    best_result = result
                    best_w = w_init
                    best_h = h_init

    # Strategy 3: High-precision polish with smooth-max
    if best_result is not None and time.time() < deadline - 8.0:
        polished = _polish(best_result, best_w, best_h, max_iter=2000)
        if _validate(polished, best_w, best_h, tol=1e-7):
            value = _total_radius(polished)
            if value > best_value:
                best_value = value
                best_result = polished

    # Strategy 4: Final micro-polish on the very best result
    if best_result is not None and time.time() < deadline - 3.0:
        # Do one more round of penalty optimization with very high beta
        micro = _local_optimize(best_result, best_w, best_h, max_iter=1000)
        if _validate(micro, best_w, best_h, tol=1e-7):
            value = _total_radius(micro)
            if value > best_value:
                best_value = value
                best_result = micro

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
