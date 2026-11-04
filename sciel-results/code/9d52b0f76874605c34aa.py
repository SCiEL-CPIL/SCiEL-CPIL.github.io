import numpy as np
from scipy.optimize import minimize
import time

N = 21

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

def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
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
                if dist_sq < (min_dist - tol) ** 2:
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

def _objective_and_grad(params, W, H):
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
    boundary_weight = 200.0

    def _smooth_max_grad(v):
        """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = -(x - r) = r - x
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

    # x + r <= W  => violation = x + r - W
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

    # Overlap penalties: vectorized computation
    overlap_weight = 200.0
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
    obj += overlap_weight * np.sum(smax_vec ** 2)
    
    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
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

def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)

def _polish(circles, W, H, max_iter=8000):
    """
    Three-pass polish: coarse, medium, fine.
    """
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 150}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)

def _hexagonal_init(W, H):
    """
    Generate a hexagonal grid initialization.
    Returns (N, 3) array of [x, y, r].
    """
    circles = []
    # Try different scales to fit 21 circles
    # Hexagonal packing: rows offset by half spacing
    # For a square-ish region, let's try a few configurations
    
    best_circles = None
    best_score = -1
    
    # Try different numbers of columns
    for n_cols in range(4, 7):
        n_rows = (N + n_cols - 1) // n_cols
        # Spacing
        # Width W, Height H
        # For hexagonal: horizontal spacing dx, vertical spacing dy = dx * sqrt(3)/2
        # To fit n_cols in W: dx = W / (n_cols + 0.5) approx
        # To fit n_rows in H: dy = H / (n_rows + 0.5) approx
        # r = min(dx, dy/sqrt(3)) / 2 approx
        
        dx = W / (n_cols + 0.5)
        dy = H / (n_rows + 0.5)
        r = min(dx, dy / (np.sqrt(3)/2)) / 2 * 0.95 # 5% safety
        
        if r <= 0:
            continue
            
        centers = []
        count = 0
        for row in range(n_rows):
            y = (row + 0.5) * dy
            offset = (row % 2) * dx / 2
            for col in range(n_cols):
                if count >= N:
                    break
                x = (col + 0.5 + offset) * dx
                # Check if inside
                if x >= r and x <= W - r and y >= r and y <= H - r:
                    centers.append([x, y, r])
                    count += 1
            if count >= N:
                break
                
        if len(centers) == N:
            circles_arr = np.array(centers)
            # Check validity
            if _check_validity(circles_arr, W, H):
                score = np.sum(circles_arr[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles_arr
                    
    if best_circles is not None:
        return best_circles
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        r = min(W, H) / side * 0.45
        centers = []
        for i in range(N):
            col = i % side
            row = i // side
            x = (col + 0.5) * W / side
            y = (row + 0.5) * H / side
            centers.append([x, y, r])
        return np.array(centers)

def _pack_for_aspect(W, H, start_time, deadline):
    """
    Pack circles in W x H rectangle.
    Returns best (pos, radii) and score found for this aspect.
    """
    best_score = 0
    best_circles = None
    
    # 1. Hexagonal Initialization
    init_circles = _hexagonal_init(W, H)
    
    # 2. Local Optimization
    try:
        opt_circles = _local_optimize(init_circles, W, H, max_iter=2000)
        # Polish
        opt_circles = _polish(opt_circles, W, H, max_iter=5000)
        
        # Check validity
        if _check_validity(opt_circles, W, H):
            score = np.sum(opt_circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = opt_circles.copy()
        else:
            # Try to repair
            repaired, valid = _separate_circles(opt_circles, W, H)
            if valid:
                score = np.sum(repaired[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = repaired.copy()
    except Exception:
        pass
    
    # 3. Random Restarts
    if time.time() < deadline - 10:
        for seed in [42, 123, 456, 789, 1011]:
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(seed)
            max_r = min(W, H) / 2
            x = rng.rand(N) * W
            y = rng.rand(N) * H
            r = rng.rand(N) * 0.1 * max_r
            rand_circles = np.column_stack([x, y, r])
            
            try:
                opt_circles = _local_optimize(rand_circles, W, H, max_iter=1500)
                opt_circles = _polish(opt_circles, W, H, max_iter=3000)
                
                if _check_validity(opt_circles, W, H):
                    score = np.sum(opt_circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = opt_circles.copy()
                else:
                    repaired, valid = _separate_circles(opt_circles, W, H)
                    if valid:
                        score = np.sum(repaired[:, 2])
                        if score > best_score:
                            best_score = score
                            best_circles = repaired.copy()
            except Exception:
                continue
                
    return best_circles, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_global_score = 0
    best_global_result = None
    
    np.random.seed(42)
    
    # Aspect ratios to try
    # Perimeter 4 => 2(W+H)=4 => W+H=2 => H = 2-W
    # W in [0.5, 1.5]
    # Let's try a range of aspect ratios W/H
    # ar = W/H
    # W = 2*ar/(1+ar), H = 2/(1+ar)
    aspect_ratios = np.linspace(0.7, 1.4, 15)
    
    # Ensure we check the square (ar=1.0)
    if not np.any(np.isclose(aspect_ratios, 1.0)):
        aspect_ratios = np.insert(aspect_ratios, len(aspect_ratios)//2, 1.0)
        
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        circles, score = _pack_for_aspect(W, H, start_time, deadline)
        
        if circles is not None and score > best_global_score:
            # Ensure strict validity
            if _check_validity(circles, W, H):
                best_global_score = score
                best_global_result = circles.copy()
            else:
                # Try to fix
                repaired, valid = _separate_circles(circles, W, H)
                if valid:
                    score = np.sum(repaired[:, 2])
                    if score > best_global_score:
                        best_global_score = score
                        best_global_result = repaired.copy()
    
    # Fallback if no valid result found
    if best_global_result is None:
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val * 0.9
        best_global_result = np.column_stack([pos, rad])
        
    return best_global_result
