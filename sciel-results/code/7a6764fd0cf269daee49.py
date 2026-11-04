"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

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

def _total_radius(circles):
    return sum(c[2] for c in circles)

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

def _make_hexagonal_grid(n, w, h):
    """Create a hexagonal packing of n circles in rectangle w x h."""
    circles = []
    row_counts = [5, 4, 5, 4, 3]
    total_rows = len(row_counts)
    
    max_cols = max(row_counts)
    r1 = w / (2 * max_cols)
    r2 = h / ((total_rows - 1) * np.sqrt(3) + 2)
    r = min(r1, r2) * 0.95
    
    if r <= 0:
        r = 0.05
    
    y_spacing = np.sqrt(3) * r
    x_spacing = 2 * r
    
    y_start = r
    for row_idx, count in enumerate(row_counts):
        y = y_start + row_idx * y_spacing
        row_width = (count - 1) * x_spacing
        x_start = (w - row_width) / 2
        for col in range(count):
            if len(circles) >= n:
                break
            x = x_start + col * x_spacing
            circles.append((x, y, r))
    
    while len(circles) < n:
        circles.append((w / 2, h / 2, r * 0.5))
    
    circles = circles[:n]
    return np.array(circles)

def _make_grid(n, w, h, cols, rows):
    """Create a rectangular grid of n circles in rectangle w x h."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            r = 0.5 / max(cols, rows) * min(w, h) * 0.95
            circles.append((cx, cy, r))
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.05))
    return np.array(circles)

def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
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
    grad[2::3] = -1.0

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
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
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
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

def _local_optimize(circles, W, H, max_iter=3000, eps=1e-4):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    try:
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps, 200.0, 200.0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return circles

def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    eps_schedule = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    n_passes = len(eps_schedule)
    iter_per_pass = max(500, max_iter // n_passes)

    result = None
    for p in range(n_passes):
        eps_val = eps_schedule[p]
        try:
            result = minimize(
                _objective_and_grad, params, args=(W, H, eps_val, 200.0, 200.0),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': iter_per_pass, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
            )
            params = result.x
        except Exception:
            break

    if result is not None:
        return result.x.reshape(N, 3)
    return circles

def _high_precision_polish(circles, W, H, max_iter=12000):
    """Ultra-tight final polish for the best candidate."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    eps_schedule = [1e-5, 1e-6, 1e-7, 1e-8]
    n_passes = len(eps_schedule)
    iter_per_pass = max(500, max_iter // n_passes)

    result = None
    for p in range(n_passes):
        eps_val = eps_schedule[p]
        try:
            result = minimize(
                _objective_and_grad, params, args=(W, H, eps_val, 500.0, 500.0),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': iter_per_pass, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 300}
            )
            params = result.x
        except Exception:
            break

    if result is not None:
        return result.x.reshape(N, 3)
    return circles

def _local_search_radius_increase(circles, W, H, max_iter=5000, tol=1e-10):
    """
    Local search: try to increase individual radii while maintaining validity.
    Uses a greedy approach with small perturbations.
    """
    circles = circles.copy()
    n = len(circles)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try increasing each radius slightly
        for i in range(n):
            for dr in [1e-4, 5e-5, 2e-5, 1e-5]:
                new_r = circles[i, 2] + dr
                # Check if new radius fits within bounds
                max_r_x = min(circles[i, 0], W - circles[i, 0])
                max_r_y = min(circles[i, 1], H - circles[i, 1])
                if new_r > max(max_r_x, max_r_y) + tol:
                    continue
                
                # Tentatively increase radius
                old_r = circles[i, 2]
                circles[i, 2] = new_r
                
                # Check validity
                if _check_validity(circles, W, H, tol=tol):
                    improved = True
                    break
                else:
                    circles[i, 2] = old_r
        
        if not improved:
            # Try small positional adjustments to create space
            for i in range(n):
                for dx in [-1e-4, 1e-4, -5e-5, 5e-5]:
                    for dy in [-1e-4, 1e-4, -5e-5, 5e-5]:
                        old_x, old_y = circles[i, 0], circles[i, 1]
                        circles[i, 0] = old_x + dx
                        circles[i, 1] = old_y + dy
                        
                        # Re-clamp to bounds
                        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
                        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
                        
                        if _check_validity(circles, W, H, tol=tol):
                            # Try to increase radius now
                            for dr in [1e-4, 5e-5, 2e-5, 1e-5]:
                                new_r = circles[i, 2] + dr
                                max_r_x = min(circles[i, 0], W - circles[i, 0])
                                max_r_y = min(circles[i, 1], H - circles[i, 1])
                                if new_r > max(max_r_x, max_r_y) + tol:
                                    continue
                                
                                old_r = circles[i, 2]
                                circles[i, 2] = new_r
                                if _check_validity(circles, W, H, tol=tol):
                                    improved = True
                                    break
                                else:
                                    circles[i, 2] = old_r
                            
                            if improved:
                                break
                        else:
                            circles[i, 0] = old_x
                            circles[i, 1] = old_y
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    return circles

def _make_random_init(n, w, h, seed=None):
    """Create a random initialization of n circles in rectangle w x h."""
    rng = np.random.RandomState(seed)
    circles = []
    for i in range(n):
        r = 0.03 + 0.04 * rng.random()
        x = rng.uniform(r, w - r)
        y = rng.uniform(r, h - r)
        circles.append((x, y, r))
    return np.array(circles)

def _evaluate_aspect_ratio(ar, start_time, deadline):
    """
    Evaluate a single aspect ratio: try multiple initializations, optimize,
    return (score, circles, W, H).
    """
    W = 2.0 * ar / (1.0 + ar)
    H = 2.0 / (1.0 + ar)
    
    best_score = 0.0
    best_circles = None
    
    # Try hexagonal grid
    hex_circles = _make_hexagonal_grid(N, W, H)
    hex_circles, valid = _separate_circles(hex_circles, W, H)
    if valid:
        score = _total_radius(hex_circles)
        if score > best_score:
            best_score = score
            best_circles = hex_circles.copy()
        
        if time.time() < deadline - 20.0:
            opt = _local_optimize(hex_circles, W, H, max_iter=3000, eps=1e-4)
            opt, valid = _separate_circles(opt, W, H)
            if valid:
                score = _total_radius(opt)
                if score > best_score:
                    best_score = score
                    best_circles = opt.copy()
    
    # Try rectangular grids
    for cols, rows in [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (5, 4), (4, 5)]:
        if time.time() > deadline - 20.0:
            break
        
        grid_circles = _make_grid(N, W, H, cols, rows)
        grid_circles, valid = _separate_circles(grid_circles, W, H)
        if valid:
            score = _total_radius(grid_circles)
            if score > best_score:
                best_score = score
                best_circles = grid_circles.copy()
            
            if time.time() < deadline - 20.0:
                opt = _local_optimize(grid_circles, W, H, max_iter=2000, eps=1e-4)
                opt, valid = _separate_circles(opt, W, H)
                if valid:
                    score = _total_radius(opt)
                    if score > best_score:
                        best_score = score
                        best_circles = opt.copy()
    
    # Try random initializations
    for seed in [42, 123, 777]:
        if time.time() > deadline - 25.0:
            break
        
        rand_circles = _make_random_init(N, W, H, seed=seed)
        rand_circles, valid = _separate_circles(rand_circles, W, H)
        if valid:
            score = _total_radius(rand_circles)
            if score > best_score:
                best_score = score
                best_circles = rand_circles.copy()
            
            if time.time() < deadline - 20.0:
                opt = _local_optimize(rand_circles, W, H, max_iter=2000, eps=1e-4)
                opt, valid = _separate_circles(opt, W, H)
                if valid:
                    score = _total_radius(opt)
                    if score > best_score:
                        best_score = score
                        best_circles = opt.copy()
    
    return best_score, best_circles, W, H

def _nelder_mead_aspect_search(start_time, deadline):
    """
    Use Nelder-Mead to optimize aspect ratio, maximizing total radius.
    Returns (best_score, best_circles, best_W, best_H).
    """
    def objective(ar):
        if time.time() > deadline - 30.0:
            return -1.0
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        # Quick evaluation with limited iterations
        hex_circles = _make_hexagonal_grid(N, W, H)
        hex_circles, valid = _separate_circles(hex_circles, W, H, max_iter=1000)
        if not valid:
            return -1.0
        score = _total_radius(hex_circles)
        # Quick local optimization
        if time.time() < deadline - 35.0:
            opt = _local_optimize(hex_circles, W, H, max_iter=1000, eps=1e-3)
            opt, valid = _separate_circles(opt, W, H, max_iter=1000)
            if valid:
                score = max(score, _total_radius(opt))
        return -score  # Negate for minimization
    
    # Initial simplex: aspect ratios from 0.5 to 2.0
    x0 = np.array([1.0, 0.8, 1.2, 0.7, 1.3])
    
    try:
        result = minimize(
            objective, x0[0], method='Nelder-Mead',
            options={'maxiter': 30, 'xatol': 0.05, 'fatol': 0.01, 'adaptive': True}
        )
        best_ar = result.x[0]
    except Exception:
        best_ar = 1.0
    
    # Evaluate the best aspect ratio found
    best_score, best_circles, best_W, best_H = _evaluate_aspect_ratio(
        best_ar, start_time, deadline
    )
    
    return best_score, best_circles, best_W, best_H

def _refine_aspect_around(best_ar, best_score, best_circles, best_W, best_H, start_time, deadline):
    """
    Refine aspect ratio around the best found value using a finer grid.
    """
    if time.time() > deadline - 30.0:
        return best_score, best_circles, best_W, best_H
    
    # Try nearby aspect ratios
    for delta in [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]:
        if time.time() > deadline - 25.0:
            break
        
        ar = best_ar + delta
        if ar <= 0.3 or ar >= 3.0:
            continue
        
        score, circles, W, H = _evaluate_aspect_ratio(ar, start_time, deadline)
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles.copy()
            best_W = W
            best_H = H
    
    return best_score, best_circles, best_W, best_H

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_circles = None
    best_score = 0.0
    best_W = 1.0
    best_H = 1.0

    # Phase 1: Quick grid search over aspect ratios
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.75, 1.25, 0.85, 1.15,
        0.65, 1.35, 0.6, 1.4, 0.95, 1.05,
    ]

    for ar in aspect_ratios:
        if time.time() > deadline - 40.0:
            break
        
        score, circles, W, H = _evaluate_aspect_ratio(ar, start_time, deadline)
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles.copy()
            best_W = W
            best_H = H

    # Phase 2: Nelder-Mead refinement of aspect ratio
    if time.time() < deadline - 50.0:
        nm_score, nm_circles, nm_W, nm_H = _nelder_mead_aspect_search(start_time, deadline)
        if nm_circles is not None and nm_score > best_score:
            best_score = nm_score
            best_circles = nm_circles.copy()
            best_W = nm_W
            best_H = nm_H

    # Phase 2.5: Fine-grained aspect ratio refinement around best
    if best_circles is not None and time.time() < deadline - 35.0:
        best_score, best_circles, best_W, best_H = _refine_aspect_around(
            best_W / best_H, best_score, best_circles, best_W, best_H, start_time, deadline
        )

    # Phase 3: Final high-precision polish
    if best_circles is not None and time.time() < deadline - 15.0:
        polished = _polish(best_circles, best_W, best_H, max_iter=10000)
        polished, valid = _separate_circles(polished, best_W, best_H)
        if valid:
            score = _total_radius(polished)
            if score > best_score:
                best_score = score
                best_circles = polished.copy()

    # Phase 4: Local search for radius increases
    if best_circles is not None and time.time() < deadline - 10.0:
        ls_result = _local_search_radius_increase(best_circles, best_W, best_H, max_iter=3000)
        ls_result, valid = _separate_circles(ls_result, best_W, best_H)
        if valid:
            score = _total_radius(ls_result)
            if score > best_score:
                best_score = score
                best_circles = ls_result.copy()

    # Phase 5: Ultra-high precision final polish
    if best_circles is not None and time.time() < deadline - 5.0:
        final = _high_precision_polish(best_circles, best_W, best_H, max_iter=8000)
        final, valid = _separate_circles(final, best_W, best_H)
        if valid:
            score = _total_radius(final)
            if score > best_score:
                best_score = score
                best_circles = final.copy()

    if best_circles is not None:
        return np.array(best_circles)

    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
