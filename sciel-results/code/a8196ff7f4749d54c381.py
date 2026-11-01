"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
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


def _compute_energy(cx, cy, r, W, H):
    """Compute surrogate energy: penalizes overlaps and wall proximity."""
    energy = 0.0
    # Wall penalties
    for i in range(N):
        energy += max(0, r[i] - cx[i]) ** 2
        energy += max(0, cx[i] + r[i] - W) ** 2
        energy += max(0, r[i] - cy[i]) ** 2
        energy += max(0, cy[i] + r[i] - H) ** 2
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rsum = r[i] + r[j]
            if d2 < rsum * rsum:
                overlap = rsum - np.sqrt(max(d2, 1e-12))
                energy += overlap * overlap
    return energy


def _grid_packing(w, h, n_circles):
    """Generate initial positions using a grid layout with uniform radii."""
    positions = []
    # Try different grid configurations
    best_config = None
    best_min_dist = 0
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < n_circles:
                continue
            cell_w = w / cols
            cell_h = h / rows
            # Max radius is half the minimum cell dimension
            r_est = min(cell_w, cell_h) / 2 * 0.95
            
            # Check if all circles fit
            valid = True
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                if row >= rows:
                    valid = False
                    break
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                if x - r_est < 0 or x + r_est > w or y - r_est < 0 or y + r_est > h:
                    valid = False
                    break
            
            if valid:
                # Compute minimum distance between any two circles
                min_d = float('inf')
                for i in range(n_circles):
                    row_i = i // cols
                    col_i = i % cols
                    xi = (col_i + 0.5) * cell_w
                    yi = (row_i + 0.5) * cell_h
                    for j in range(i + 1, n_circles):
                        row_j = j // cols
                        col_j = j % cols
                        xj = (col_j + 0.5) * cell_w
                        yj = (row_j + 0.5) * cell_h
                        d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        min_d = min(min_d, d)
                
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_config = (rows, cols, r_est)
    
    if best_config is None:
        # Fallback: single row
        cols = n_circles
        cell_w = w / cols
        r_est = cell_w / 2 * 0.9
        for i in range(n_circles):
            x = (i + 0.5) * cell_w
            y = h / 2
            positions.append((x, y, r_est))
    else:
        rows, cols, r_est = best_config
        cell_w = w / cols
        cell_h = h / rows
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * cell_w
            y = (row + 0.5) * cell_h
            positions.append((x, y, r_est))
    
    return positions[:n_circles]


def _hexagonal_packing(w, h, n_circles):
    """Generate initial positions using hexagonal close packing."""
    positions = []
    # Estimate radius for hexagonal packing
    # Hexagonal packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area per circle = w*h / n
    # pi * r^2 = (w*h/n) / 0.9069
    r_est = np.sqrt((w * h / n_circles) * (2 * np.sqrt(3) / np.pi))
    r_est = min(r_est * 0.95, w / 6, h / 6)
    
    if r_est < 1e-6:
        return []

    # Determine rows and cols
    row_height = r_est * np.sqrt(3)
    col_width = r_est * 2
    
    max_rows = int(h / row_height) + 1
    max_cols = int(w / col_width) + 1
    
    # Generate candidate positions
    candidates = []
    for row in range(max_rows):
        for col in range(max_cols):
            x = (col + 0.5) * col_width + (row % 2) * r_est
            y = (row + 0.5) * row_height
            if x + r_est <= w + 1e-6 and y + r_est <= h + 1e-6:
                candidates.append((x, y))
    
    # If we have enough candidates, pick the first n_circles
    if len(candidates) >= n_circles:
        for i in range(n_circles):
            x, y = candidates[i]
            positions.append((x, y, r_est))
        return positions
    
    # If not enough, we need to adjust radius or accept fewer
    # For now, just return what we have, construct() will handle fallbacks or dynamics will fix it
    # But better to try a smaller radius if we can fit more
    if len(candidates) > 0:
        # Try to fit n_circles by reducing radius slightly
        # This is a heuristic
        for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
            r_try = r_est * scale
            row_height_try = r_try * np.sqrt(3)
            col_width_try = r_try * 2
            max_rows_try = int(h / row_height_try) + 1
            max_cols_try = int(w / col_width_try) + 1
            candidates_try = []
            for row in range(max_rows_try):
                for col in range(max_cols_try):
                    x = (col + 0.5) * col_width_try + (row % 2) * r_try
                    y = (row + 0.5) * row_height_try
                    if x + r_try <= w + 1e-6 and y + r_try <= h + 1e-6:
                        candidates_try.append((x, y))
            if len(candidates_try) >= n_circles:
                for i in range(n_circles):
                    x, y = candidates_try[i]
                    positions.append((x, y, r_try))
                return positions
    
    return positions[:n_circles]


def _dynamics_simulate(x, y, r, w, h, time_budget, dt=0.002, max_iter=20000):
    """
    Rigid-body dynamics simulation with impulse-based collision resolution.
    Mass proportional to radius squared.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.2 or h < 0.2:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-6)

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    start_time = time.time()
    
    # Adaptive force scaling
    wall_stiffness = 8000.0
    pair_stiffness = 5000.0

    iu = np.triu_indices(N, k=1)

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Anneal damping: start high to settle, end low to allow movement
        progress = it / max_iter
        damping = 0.98 - 0.05 * progress # 0.98 to 0.93
        
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_stiffness
        fx += right_viol * wall_stiffness
        fy -= bottom_viol * wall_stiffness
        fy += top_viol * wall_stiffness

        # Pairwise repulsion - vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        if np.max(overlap) > 1e-10:
            # Force is proportional to overlap
            f_pairs = overlap * pair_stiffness
            
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            # Vectorized force accumulation using np.add.at
            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        # Update velocities
        vx += dt * fx / mass
        vy += dt * fy / mass

        # Damping
        vx *= damping
        vy *= damping

        # Update positions
        x += dt * vx
        y += dt * vy

        # Clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Impulse-based collision resolution (post-step correction) - multiple passes
        n_passes = 10
        
        for _ in range(n_passes):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) <= 1e-9:
                break

            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            m_i = mass[iu[0]]
            m_j = mass[iu[1]]
            total_m = m_i + m_j
            correction = overlap * 0.5
            frac_i = m_j / total_m
            frac_j = m_i / total_m

            dx_corr = ux * correction * frac_i
            dy_corr = uy * correction * frac_i
            np.add.at(x, iu[0], dx_corr)
            np.add.at(y, iu[0], dy_corr)
            np.add.at(x, iu[1], -ux * correction * frac_j)
            np.add.at(y, iu[1], -uy * correction * frac_j)

            # Re-clamp after correction
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Iteratively expand radii.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    
    # Phase 1: Uniform scaling
    # Try to scale all radii by a factor
    for _ in range(200):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            # Binary search for the max uniform scale
            lo = 1.0
            hi = 1.005
            for _ in range(20):
                mid = (lo + hi) / 2.0
                r_test = r * mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            r = r * lo
            break

    # Phase 2: Individual expansion
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            # Max radius limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed

            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid

                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_aspect(x, y, r, w0, h0, time_budget):
    """
    Optimize aspect ratio (width) by trying different widths.
    Returns (best_w, best_x, best_y, best_r, best_h)
    """
    best_w = w0
    best_h = h0
    best_val = np.sum(r)
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    start_time = time.time()
    
    # Coarse search
    for w in np.linspace(0.4, 1.6, 40):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        # Scale positions to fit new rectangle
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        
        # Quick validity check
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w
                best_h = h
                best_x = x_new.copy()
                best_y = y_new.copy()
                best_r = r_new.copy()
    
    # Fine search around best
    for w in np.linspace(max(0.3, best_w - 0.2), min(1.7, best_w + 0.2), 30):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
            
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w
                best_h = h
                best_x = x_new.copy()
                best_y = y_new.copy()
                best_r = r_new.copy()
                
    return best_w, best_x, best_y, best_r, best_h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate initial guesses using multiple strategies
    def get_initial_guesses():
        guesses = []
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.25, 0.7, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05, 0.88, 1.12, 0.72, 1.28, 0.65, 1.35, 0.58, 1.42]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            # Try hexagonal packing
            positions = _hexagonal_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
            
            # Try grid packing
            positions = _grid_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
            
            # Try corner-anchored packing
            # Place 4 circles in corners, then fill center
            r_corner = min(w0, h0) * 0.25
            x_c = np.array([r_corner, w0-r_corner, r_corner, w0-r_corner])
            y_c = np.array([r_corner, r_corner, h0-r_corner, h0-r_corner])
            r_c = np.full(4, r_corner)
            
            # Fill remaining with grid in center
            remaining = N - 4
            if remaining > 0:
                # Grid in the central region
                cx_min = 2*r_corner
                cx_max = w0 - 2*r_corner
                cy_min = 2*r_corner
                cy_max = h0 - 2*r_corner
                if cx_max > cx_min and cy_max > cy_min:
                    grid_positions = _grid_packing(cx_max - cx_min, cy_max - cy_min, remaining)
                    if len(grid_positions) >= remaining:
                        x_g = np.array([p[0] + cx_min for p in grid_positions[:remaining]])
                        y_g = np.array([p[1] + cy_min for p in grid_positions[:remaining]])
                        r_g = np.array([p[2] for p in grid_positions[:remaining]])
                        x = np.concatenate([x_c, x_g])
                        y = np.concatenate([y_c, y_g])
                        r = np.concatenate([r_c, r_g])
                        guesses.append((x, y, r, w0, h0))

        # Random starts with varied radii
        rng = np.random.default_rng(42)
        for trial in range(50):
            w0 = rng.uniform(0.45, 1.55)
            h0 = 2.0 - w0
            x = rng.uniform(0.06, w0 - 0.06, N)
            y = rng.uniform(0.06, h0 - 0.06, N)
            # Vary radius range per trial
            r_min = rng.uniform(0.03, 0.07)
            r_max = rng.uniform(0.09, 0.15)
            r = rng.uniform(r_min, r_max, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by initial energy (lower is better)
    def guess_energy(g):
        x, y, r, w, h = g
        return _compute_energy(x, y, r, w, h)
    
    guesses.sort(key=guess_energy)

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Step 1: Dynamics simulation to resolve overlaps
        remaining = deadline - (time.time() - start_time)
        sim_time = min(3.0, remaining * 0.2)
        x_sim, y_sim, r_sim = _dynamics_simulate(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        remaining = deadline - (time.time() - start_time)
        aspect_time = min(1.5, remaining * 0.1)
        w_opt, x_opt, y_opt, r_opt, h_opt = _optimize_aspect(x_sim, y_sim, r_sim, w0, h0, aspect_time)
        
        if abs(w_opt - w0) > 1e-6:
            # Re-run short dynamics to allow rearrangement after scaling
            sim_time_short = min(1.0, remaining * 0.05)
            x_re, y_re, r_re = _dynamics_simulate(x_opt, y_opt, r_opt, w_opt, h_opt, sim_time_short)
            if x_re is not None:
                x_sim, y_sim, r_sim = x_re, y_re, r_re
                w0, h0 = w_opt, h_opt

        # Step 3: Expand radii
        remaining = deadline - (time.time() - start_time)
        expand_time = min(10.0, remaining * 0.5)
        x_exp, y_exp, r_exp = _expand_radii(x_sim, y_sim, r_sim, w0, h0, expand_time)
        if x_exp is not None:
            valid, val = check_validity(x_exp, y_exp, r_exp, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w0, h0)
        else:
            valid, val = check_validity(x_sim, y_sim, r_sim, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_sim.copy(), y_sim.copy(), r_sim.copy(), w0, h0)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 2:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = max(1.0, deadline - (time.time() - start_time) - 1)
        x_final, y_final, r_final = _expand_radii(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)

    if best_result is None:
        # Fallback: simple grid in a 1x1 square
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x_pos = (i + 0.5) / side
                y_pos = (j + 0.5) / side
                r_pos = 0.5 / side - 0.005
                cells.append((x_pos, y_pos, r_pos))
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
