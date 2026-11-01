"""Circle packing in perimeter-4 rectangle: Direct optimization with smooth penalties."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _smooth_penalty(overlaps, violations, scale=50.0):
    """
    Compute a smooth penalty using log-sum-exp approximation of max.
    log(1 + sum(exp(scale * v))) approximates max(0, v) * scale for large scale.
    """
    # Clip to avoid overflow in exp
    overlaps = np.clip(overlaps, -10, 10)
    violations = np.clip(violations, -10, 10)
    
    # log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    def logsumexp(arr):
        if len(arr) == 0:
            return 0.0
        m = np.max(arr)
        if m > 50:
            return m + np.log(len(arr))
        return np.log(1.0 + np.sum(np.exp(arr)))

    pen_overlap = logsumexp(scale * overlaps)
    pen_viol = logsumexp(scale * violations)
    
    return (pen_overlap + pen_viol) / scale


def _objective(z, scale=50.0):
    """
    Objective function for optimization.
    z = [x_1..x_N, y_1..y_N, r_1..r_N, w]
    h = 2 - w
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    w = z[3*N]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only compute for i < j to save time and avoid double counting
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlaps = r_sum[iu] - d_pairs
    
    # Boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    
    violations = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    
    penalty = _smooth_penalty(overlaps, violations, scale)
    
    # Add penalty for aspect ratio extremes if needed, but perimeter constraint is handled by w+h=2
    # We might want to discourage very thin rectangles
    if w < 0.3:
        penalty += (0.3 - w) ** 2 * 10
    elif w > 1.7:
        penalty += (w - 1.7) ** 2 * 10

    return obj + penalty


def _force_relax(x, y, r, w, h, max_iter=500, dt_init=0.002):
    """Force-directed relaxation to fix any remaining small overlaps/violations."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-8:
            break

        f_pairs = overlap * 10.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

    return x, y


def _grow_radii(x, y, r, w, h, time_budget, steps=100):
    """Grow all radii uniformly as much as possible."""
    for _ in range(steps):
        if time.time() > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    return r


def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually."""
    n = N
    for i in range(n):
        if time.time() > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-6:
            r[i] = r_max
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Aspect ratios to try
    # Perimeter 4 => w + h = 2.
    # Square: w=1, h=1
    # Rectangles: w=0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.55, 1.45]

    # Initial grid configurations (rows, cols)
    # 21 circles.
    # 3x7, 7x3
    # 4x6 (24 cells, 3 empty), 6x4
    # 5x5 (25 cells, 4 empty)
    grid_configs = [
        (3, 7), (7, 3),
        (4, 6), (6, 4),
        (5, 5),
        (3, 8), (8, 3), # Just in case
        (2, 11), (11, 2) # Very flat
    ]

    # We will try to optimize from multiple starts
    # To save time, we limit the number of optimization runs
    max_optimizations = 15
    opt_count = 0

    for w0 in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Find a good grid for this aspect ratio
        # The grid aspect ratio should match the container aspect ratio roughly
        # w0/h0 ~= cols/rows
        best_grid = None
        best_diff = 1e9
        for r, c in grid_configs:
            if r * c < N:
                continue
            grid_aspect = c / r
            container_aspect = w0 / h0
            diff = abs(grid_aspect - container_aspect)
            if diff < best_diff:
                best_diff = diff
                best_grid = (r, c)

        if best_grid is None:
            best_grid = (5, 5) # Fallback

        rows, cols = best_grid
        
        # Create initial grid layout
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w0 / cols
        cell_h = h0 / rows
        base_r = min(cell_w, cell_h) * 0.45

        for i in range(N):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) * cell_w
            y[i] = (row + 0.5) * cell_h
            r[i] = base_r

        # Add small random perturbation to break symmetry
        np.random.seed(42 + int(w0 * 100))
        x += np.random.uniform(-0.01, 0.01, N)
        y += np.random.uniform(-0.01, 0.01, N)
        
        # Ensure initial validity (force relax)
        x, y = _force_relax(x, y, r, w0, h0, max_iter=100)
        
        if not _is_valid(x, y, r, w0, h0):
            continue

        # Construct initial z vector
        z0 = np.concatenate([x, y, r, [w0]])
        
        # Define bounds for L-BFGS-B
        # x, y can be large, but we'll use penalty. 
        # r must be > 0.
        # w must be in [0.1, 1.9]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((0.001, 1.0)) # r
        bounds.append((0.1, 1.9)) # w

        # Run optimization
        try:
            res = minimize(
                _objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
            )
        except Exception:
            continue

        if res.success or res.fun < 1e6:
            z_opt = res.x
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            r_opt = z_opt[2*N:3*N]
            w_opt = z_opt[3*N]
            h_opt = 2.0 - w_opt

            # Post-processing: Force relax to fix numerical issues
            x_opt, y_opt = _force_relax(x_opt, y_opt, r_opt, w_opt, h_opt, max_iter=200)
            
            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                # Grow radii
                time_budget = min(deadline - 2, time.time() + 2.0)
                r_grown = _grow_radii(x_opt, y_opt, r_opt, w_opt, h_opt, time_budget)
                r_expanded = _expand_individual(x_opt, y_opt, r_grown, w_opt, h_opt, time_budget)
                
                if _is_valid(x_opt, y_opt, r_expanded, w_opt, h_opt):
                    value = np.sum(r_expanded)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([x_opt, y_opt, r_expanded])
                        opt_count += 1
                        if opt_count >= max_optimizations:
                            break
        else:
            # If optimization failed, maybe just use the relaxed initial guess
            if _is_valid(x, y, r, w0, h0):
                time_budget = min(deadline - 2, time.time() + 1.0)
                r_grown = _grow_radii(x, y, r, w0, h0, time_budget)
                r_expanded = _expand_individual(x, y, r_grown, w0, h0, time_budget)
                if _is_valid(x, y, r_expanded, w0, h0):
                    value = np.sum(r_expanded)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([x, y, r_expanded])

    # Fallback if no valid result found
    if best_result is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Ensure perimeter constraint for fallback
        # The fallback grid is in [0,1]x[0,1], perimeter 4.
        # But we need to check if it's valid.
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        if not _is_valid(x, y, r, 1.0, 1.0):
             # Adjust radii to be smaller
             r *= 0.9
             best_result[:, 2] = r

    return best_result
