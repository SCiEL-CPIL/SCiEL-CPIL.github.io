"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 138.0  # safety margin before 150s kill


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


def _separate_circles(circles, W, H, max_iter=3000, tol=1e-12):
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

        # Resolve boundary violations by clamping
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
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
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

        # Re-clamp after separation, shrinking radius if circle doesn't fit
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


def _soft_max(v, scale):
    """Soft max of v and 0 using log-sum-exp: scale * log(1 + exp(v/scale))."""
    v_safe = np.clip(v / scale, -50, 50)
    smax = scale * np.log1p(np.exp(v_safe))
    sig = 1.0 / (1.0 + np.exp(-v_safe))
    return smax, sig


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
    """Use L-BFGS-B with analytical gradients. Two-stage: coarse then fine."""
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

    for k in range(len(eps_schedule)):
        eps_k = eps_schedule[k]
        w_k = weight_schedule[k]
        iters_k = max(500, max_iter // (len(eps_schedule) * (len(eps_schedule) - k)))
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_k, w_k, w_k),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iters_k, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
        )
        params = result.x

    return params.reshape(N, 3)


def _hex_initialization(W, H, n_circles=21):
    """Create a hexagonal grid initial configuration."""
    # Hexagonal packing: rows offset by half spacing
    # Try to fit n_circles in a hex pattern
    # For 21 circles: 5-4-5-4-4 or similar arrangements
    # Let's try a 5-column layout with hexagonal offsets
    
    # Calculate spacing to fit in rectangle
    # Hex spacing: horizontal = 2r, vertical = sqrt(3)*r, row offset = r
    # For a 5-4-5-4-3 arrangement (21 circles):
    # Actually let's try rows of 5,5,5,5,1 or 4,5,5,5,2 or 5,4,5,4,3
    
    # Let's use a more systematic approach:
    # Try different row configurations
    best_circles = None
    best_score = -np.inf
    
    row_configs = [
        [5, 5, 5, 4, 2],
        [5, 4, 5, 4, 3],
        [4, 5, 5, 5, 2],
        [5, 5, 4, 4, 3],
        [4, 4, 5, 4, 4],
        [5, 5, 5, 5, 1],
        [6, 5, 5, 3, 2],
        [5, 6, 4, 3, 3],
        [4, 5, 6, 3, 3],
        [5, 5, 6, 2, 3],
    ]
    
    for config in row_configs:
        if sum(config) != n_circles:
            continue
        
        n_rows = len(config)
        max_cols = max(config)
        
        # For hex packing: 
        # horizontal spacing = 2*r
        # vertical spacing = sqrt(3)*r
        # Each row offset by r horizontally
        
        # Total width needed: (max_cols - 1) * 2r + 2r + r (for offset) = (2*max_cols)*r
        # Actually: width = (max_cols - 1) * 2r + 2r = 2*max_cols*r
        # But with offset, the widest row extends by r on each side
        # Let's be more careful:
        # Row 0: centers at r, 3r, 5r, ... (for cols in row 0)
        # Row 1: centers at 2r, 4r, 6r, ... (offset by r)
        # Max x extent: (max_cols - 1) * 2r + 2r for non-offset rows
        # For offset rows: (max_cols - 1) * 2r + r + r = (max_cols - 1) * 2r + 2r
        # Actually the rightmost point is at x_center + r
        # For non-offset row with c columns: rightmost = r + (c-1)*2r + r = 2c*r
        # For offset row with c columns: rightmost = 2r + (c-1)*2r + r = (2c+1)*r
        # Hmm, let me just compute it directly
        
        # Try to find r that fits
        # Width constraint: max over all rows of (rightmost_x) <= W
        # Height constraint: (n_rows - 1) * sqrt(3) * r + 2r <= H
        
        # For each row, the x positions are:
        # Even rows (0-indexed): x = r + 2r*k for k=0..(c-1)
        # Odd rows: x = 2r + 2r*k for k=0..(c-1)
        
        # Max right extent:
        max_right = 0
        for row_idx, c in enumerate(config):
            if c == 0:
                continue
            if row_idx % 2 == 0:
                rightmost = r + 2*r*(c-1) + r  # = 2c*r
            else:
                rightmost = 2*r + 2*r*(c-1) + r  # = (2c+1)*r
            max_right = max(max_right, rightmost)
        
        # Width constraint: max_right <= W
        # Height constraint: (n_rows-1)*sqrt(3)*r + 2r <= H
        
        r_from_width = W / max_right if max_right > 0 else np.inf
        r_from_height = H / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r_from_width, r_from_height) * 0.99  # small margin
        
        if r <= 0 or r < 0.01:
            continue
        
        # Generate circles
        circles = []
        for row_idx, c in enumerate(config):
            y = r + row_idx * np.sqrt(3) * r
            for k in range(c):
                if row_idx % 2 == 0:
                    x = r + 2*r*k
                else:
                    x = 2*r + 2*r*k
                circles.append([x, y, r])
        
        if len(circles) == n_circles:
            circles_arr = np.array(circles)
            # Check if all circles fit
            valid = _check_validity(circles_arr, W, H)
            if valid:
                score = np.sum(circles_arr[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles_arr
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n_circles)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n_circles:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:n_circles])


def _random_initialization(W, H, rng):
    """Create a random initial configuration with moderate circles."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _best_for_aspect(W, H, rng, time_budget=20.0):
    """Find best packing for given W, H within time budget."""
    start = time.time()
    best_val = -np.inf
    best_circles = None

    # Start with hexagonal initialization
    circles_hex = _hex_initialization(W, H, N)
    
    # Local optimize hex init
    circles_opt = _local_optimize(circles_hex, W, H, max_iter=1500)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid = _separate_circles(circles_opt, W, H)
    if valid:
        s = np.sum(circles_opt[:, 2])
        if s > best_val:
            best_val = s
            best_circles = circles_opt.copy()

    # Try grid initialization too
    if time.time() - start < time_budget * 0.4:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                cells.append([cx, cy, r])
        circles_grid = np.array(cells[:N])
        circles_grid_opt = _local_optimize(circles_grid, W, H, max_iter=1000)
        circles_grid_opt[:, 0] = np.clip(circles_grid_opt[:, 0], circles_grid_opt[:, 2], W - circles_grid_opt[:, 2])
        circles_grid_opt[:, 1] = np.clip(circles_grid_opt[:, 1], circles_grid_opt[:, 2], H - circles_grid_opt[:, 2])
        circles_grid_opt[:, 2] = np.clip(circles_grid_opt[:, 2], 0.001, min(W, H) / 2)
        circles_grid_opt, valid_grid = _separate_circles(circles_grid_opt, W, H)
        if valid_grid:
            s = np.sum(circles_grid_opt[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circles_grid_opt.copy()

    # Polish the best result so far
    if best_circles is not None and time.time() - start < time_budget * 0.8:
        polished = _polish(best_circles, W, H, max_iter=3000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_pol = _separate_circles(polished, W, H)
        if valid_pol:
            s2 = np.sum(polished[:, 2])
            if s2 > best_val:
                best_val = s2
                best_circles = polished.copy()

    # Random restarts
    remaining = time_budget - (time.time() - start)
    if remaining > 3 and best_circles is not None:
        n_restarts = max(1, int(remaining / 4))
        for _ in range(n_restarts):
            if time.time() - start > time_budget - 2:
                break
            circ_rand = _random_initialization(W, H, rng)
            circ_opt = _local_optimize(circ_rand, W, H, max_iter=600)
            circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
            circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
            circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
            circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
            if valid_rand:
                s = np.sum(circ_opt[:, 2])
                if s > best_val:
                    best_val = s
                    best_circles = circ_opt.copy()

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        # Allocate time based on promise
        if 0.95 <= ar <= 1.05:
            time_budget = 22.0
        elif 0.9 <= ar <= 1.1:
            time_budget = 18.0
        elif 0.85 <= ar <= 1.2:
            time_budget = 14.0
        else:
            time_budget = 10.0

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_for_aspect(W, H, rng, time_budget=time_budget)
        if circles is not None and score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished2 = _polish(best_result, best_W, best_H, max_iter=4000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=2000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Fallback: simple grid in square
    if best_result is None:
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

    # Ensure validity
    if not _check_validity(best_result, best_W, best_H):
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
