"""Circle packing in perimeter-4 rectangle: grid-based cellular decomposition + refinement."""
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


def _force_relax(x, y, r, w, h, max_iter=1500, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
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
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

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

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 12.0
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _grow_radii(x, y, r, w, h, time_budget, steps=300):
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
    """Try to expand each circle individually by moving it to free space."""
    n = N
    for i in range(n):
        if time.time() > time_budget:
            break
        # Find max possible radius at current position
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


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()

    max_iter = int(min(3000, time_budget * 100))
    dt = 0.001
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
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
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    r = _grow_radii(x, y, r, w, h, time_budget)
    r = _expand_individual(x, y, r, w, h, time_budget)

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _grid_layout(w, h, rows, cols, time_budget):
    """Create a grid-based initial layout and refine it."""
    # Place N circles in a rows x cols grid (filling row by row)
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    cell_w = w / cols
    cell_h = h / rows
    base_r = min(cell_w, cell_h) * 0.45

    for i in range(N):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * cell_w
        y[i] = (row + 0.5) * cell_h
        r[i] = base_r

    # Force relaxation to resolve any overlaps and push to boundaries
    x, y = _force_relax(x, y, r, w, h, max_iter=800)

    if not _is_valid(x, y, r, w, h):
        return None

    # Polish and grow
    result = _polish_positions(x, y, r, w, h, time_budget)
    if result is not None:
        return result
    return None


def _optimize_aspect(w0, h0, x0, y0, r0, time_budget):
    """Optimize the aspect ratio (width) given fixed circle positions/radii pattern."""
    # We optimize w (width) with h = 2 - w
    # Objective: maximize sum of radii after relaxation

    def objective(z):
        w = z[0]
        h = 2.0 - w
        if w < 0.3 or h < 0.3:
            return 1e8
        # Scale positions and radii to fit in [0,w] x [0,h]
        # Actually, we want to keep the relative layout but adjust for the new rectangle
        # For simplicity, just check validity and return negative sum of radii
        if not _is_valid(x0 * (w / 1.0), y0 * (h / 1.0), r0, w, h):
            return 1e8
        return -np.sum(r0)

    # This is a simplification - in practice we'd need to re-relax
    # For now, just try a range of aspect ratios
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_value = 0.0
    best_result = None

    # Strategy 1: Grid-based cellular decomposition with various configurations
    # For a grid of r rows and c cols, we need r*c >= N
    # The inradius of each cell is min(w/c, h/r)/2
    # Sum of radii = N * min(w/c, h/r) / 2
    # Subject to w + h = 2

    # Enumerate grid configurations
    grid_configs = []
    for r in range(3, 9):
        for c in range(3, 9):
            if r * c >= N:
                # Optimal aspect ratio for this grid: w/c = h/r => w = 2c/(r+c), h = 2r/(r+c)
                w_opt = 2.0 * c / (r + c)
                h_opt = 2.0 * r / (r + c)
                # Try this and nearby aspect ratios
                for aspect_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    w_try = w_opt * aspect_mult
                    h_try = 2.0 - w_try
                    if h_try < 0.3 or w_try < 0.3:
                        continue
                    grid_configs.append((r, c, w_try, h_try))

    # Also add some standard aspect ratios
    standard_aspects = [
        (1.0, 1.0), (0.8, 1.2), (1.2, 0.8), (0.7, 1.3), (1.3, 0.7),
        (0.6, 1.4), (1.4, 0.6), (0.55, 1.45), (1.45, 0.55)
    ]
    for w_s, h_s in standard_aspects:
        # Find best grid for this aspect
        for r in range(3, 8):
            c = int(np.ceil(N / r))
            if r * c >= N:
                grid_configs.append((r, c, w_s, h_s))

    # Deduplicate and limit
    seen = set()
    unique_configs = []
    for config in grid_configs:
        key = (config[0], config[1], round(config[2], 2))
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)

    # Sort by expected value (higher inradius first)
    def expected_value(config):
        r, c, w, h = config
        cell_w = w / c
        cell_h = h / r
        inradius = min(cell_w, cell_h) / 2.0
        return N * inradius

    unique_configs.sort(key=expected_value, reverse=True)

    # Try each configuration with time budget
    for r, c, w, h in unique_configs:
        if time.time() > deadline - 5:
            break

        time_budget = min(deadline - 3, time.time() + 8.0)
        result = _grid_layout(w, h, r, c, time_budget)

        if result is not None:
            x, y, rad = result
            if _is_valid(x, y, rad, w, h):
                value = np.sum(rad)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x, y, rad])

    # Strategy 2: If we have time, try to further optimize the best result
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[:, 0].copy()
        y = best_result[:, 1].copy()
        r = best_result[:, 2].copy()

        # Determine the rectangle from the best result
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        bw = max_x - min_x
        bh = max_y - min_y

        # Try to grow radii further
        time_budget = min(deadline - 2, time.time() + 15.0)
        r_grown = _grow_radii(x, y, r, bw, bh, time_budget, steps=500)
        r_expanded = _expand_individual(x, y, r_grown, bw, bh, time_budget)

        if _is_valid(x, y, r_expanded, bw, bh):
            value = np.sum(r_expanded)
            if value > best_value:
                best_value = value
                best_result = np.column_stack([x, y, r_expanded])

    # Final validation
    if best_result is not None:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]

        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        bw = max_x - min_x
        bh = max_y - min_y
        perimeter = 2 * (bw + bh)

        if perimeter > 4.0 + 1e-6 or not _is_valid(x, y, r, bw, bh):
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
