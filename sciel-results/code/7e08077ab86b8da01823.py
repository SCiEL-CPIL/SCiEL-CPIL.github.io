"""Circle packing in perimeter-4 rectangle: Pure force-directed gradient descent."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
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


def _compute_forces(x, y, r, w, h, repel_strength=10.0, wall_strength=50.0,
                    center_pull=0.01, radius_growth=0.05):
    """Compute forces for force-directed system.
    
    Returns: fx, fy (position forces), fr (radius growth forces)
    """
    n = N
    fx = np.zeros(n)
    fy = np.zeros(n)
    fr = np.zeros(n)

    # --- Pairwise repulsion (overlap penalty gradient) ---
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    d_sq = np.maximum(d_sq, 1e-12)
    d = np.sqrt(d_sq)
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(n, k=1)
    d_pairs = d[iu]
    r_sum_pairs = r_sum[iu]
    dx_pairs = dx[iu]
    dy_pairs = dy[iu]

    # Overlap: positive when circles overlap
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    if np.any(overlap > 0):
        # Force magnitude proportional to overlap (smooth)
        f_mag = repel_strength * overlap
        # Unit vectors
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_mag[idx]
            fy[i_idx] += uy[idx] * f_mag[idx]
            fx[j_idx] -= ux[idx] * f_mag[idx]
            fy[j_idx] -= uy[idx] * f_mag[idx]

        # Radius growth is blocked by overlap: reduce growth for overlapping circles
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fr[i_idx] -= overlap[idx] * 2.0
            fr[j_idx] -= overlap[idx] * 2.0

    # --- Wall repulsion ---
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    fx -= wall_strength * left_viol
    fx += wall_strength * right_viol
    fy -= wall_strength * bottom_viol
    fy += wall_strength * top_viol

    # Wall violations also block radius growth
    fr -= wall_strength * (left_viol + right_viol + bottom_viol + top_viol) * 0.1

    # --- Gentle center attraction to prevent drift ---
    cx_center = w / 2.0
    cy_center = h / 2.0
    fx += center_pull * (cx_center - x)
    fy += center_pull * (cy_center - y)

    # --- Radius growth force (the objective: maximize sum of radii) ---
    # Growth is limited by distance to nearest constraint
    for i in range(n):
        # Distance to walls
        d_wall = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
        d_wall = max(d_wall, 0.0)
        
        # Distance to other circles
        d_min_circle = float('inf')
        for j in range(n):
            if i == j:
                continue
            dx_ij = x[i] - x[j]
            dy_ij = y[i] - y[j]
            d_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
            gap = d_ij - r[i] - r[j]
            if gap < d_min_circle:
                d_min_circle = gap
        d_min_circle = max(d_min_circle, 0.0)

        # Growth rate proportional to available space
        available = min(d_wall, d_min_circle)
        fr[i] += radius_growth * available

    return fx, fy, fr


def _force_directed_solve(x0, y0, r0, w0, h0, time_budget, 
                           repel_strength=10.0, wall_strength=50.0,
                           center_pull=0.01, radius_growth=0.05):
    """Run force-directed gradient descent.
    
    Returns (x, y, r, w, h, sum_r) if valid, else None.
    """
    start_time = time.time()
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    w = w0
    h = h0
    n = N

    # Adaptive learning rate
    lr = 0.002
    best_sum_r = float(np.sum(r))
    best_state = (x.copy(), y.copy(), r.copy(), w, h)
    patience = 0
    max_patience = 2000

    for it in range(50000):
        if time.time() > start_time + time_budget:
            break

        fx, fy, fr = _compute_forces(x, y, r, w, h,
                                       repel_strength, wall_strength,
                                       center_pull, radius_growth)

        # Adaptive step size based on force magnitude
        max_f = max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fr)), 1e-8)
        if max_f > 1.0:
            lr = max(0.0001, lr * 0.8)
        elif max_f < 0.01:
            lr = min(0.005, lr * 1.05)

        # Update positions
        x += lr * fx
        y += lr * fy

        # Update radii (only grow, never shrink below current)
        r_new = r + lr * fr
        r_new = np.maximum(r_new, r * 0.99)  # Allow tiny shrink for stability
        r_new = np.maximum(r_new, 0.001)
        r = r_new

        # Clamp positions to valid range
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Check validity and track best
        if _is_valid(x, y, r, w, h):
            s = float(np.sum(r))
            if s > best_sum_r:
                best_sum_r = s
                best_state = (x.copy(), y.copy(), r.copy(), w, h)
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
        else:
            # Invalid: revert to best valid state
            x, y, r, w, h = best_state
            patience += 1
            if patience > max_patience:
                break

    # Final: try to grow radii uniformly
    bx, by, br, bw, bh = best_state
    for _ in range(200):
        if time.time() > start_time + time_budget:
            break
        r_trial = br * 1.0005
        if _is_valid(bx, by, r_trial, bw, bh):
            br = r_trial
        else:
            break

    if _is_valid(bx, by, br, bw, bh):
        return bx, by, br, bw, bh, float(np.sum(br))
    return None


def _expand_single_circle(x, y, r, w, h, i, time_budget):
    """Try to expand circle i as much as possible, possibly moving it."""
    start_time = time.time()
    n = N

    # Calculate max radius for circle i at current position
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(n):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    r_max = max(0.0, r_max)

    if r_max <= r[i] + 1e-8:
        return False

    # Try to move circle i to a better position for larger radius
    # Sample positions on a grid around current position
    best_r = r[i]
    best_x, best_y = x[i], y[i]

    # Radial search: try moving in 8 directions
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)]
    
    for dx_dir, dy_dir in directions:
        for step in [0.01, 0.02, 0.03, 0.05]:
            if time.time() > start_time + time_budget:
                return False
            new_x = x[i] + dx_dir * step
            new_y = y[i] + dy_dir * step

            # Check if this position allows larger radius
            r_wall_new = min(new_x, w - new_x, new_y, h - new_y)
            r_max_new = r_wall_new
            for j in range(n):
                if i == j:
                    continue
                dx_ij = new_x - x[j]
                dy_ij = new_y - y[j]
                d_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
                r_allowed = d_ij - r[j]
                if r_allowed < r_max_new:
                    r_max_new = r_allowed
            r_max_new = max(0.0, r_max_new)

            if r_max_new > best_r + 1e-6:
                # Verify the move is valid at current radius
                valid = True
                if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                    valid = False
                else:
                    for j in range(n):
                        if i == j:
                            continue
                        dx_ij = new_x - x[j]
                        dy_ij = new_y - y[j]
                        if dx_ij * dx_ij + dy_ij * dy_ij < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                if valid:
                    best_r = r_max_new
                    best_x, best_y = new_x, new_y

    # Apply the best move and expand
    if best_x != x[i] or best_y != y[i]:
        x[i] = best_x
        y[i] = best_y
    if best_r > r[i] + 1e-8:
        r[i] = best_r
        return True
    return False


def _global_expansion(x, y, r, w, h, time_budget):
    """Iteratively expand circles one at a time."""
    start_time = time.time()
    n = N
    improved = True
    rounds = 0
    max_rounds = 50

    while improved and rounds < max_rounds:
        if time.time() > start_time + time_budget:
            break
        improved = False
        rounds += 1
        for i in range(n):
            if time.time() > start_time + time_budget:
                break
            if _expand_single_circle(x, y, r, w, h, i, 0.5):
                improved = True

    # Final uniform growth
    for _ in range(300):
        if time.time() > start_time + time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r, w, h, float(np.sum(r))
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Generate diverse initial configurations ---
    configs = []

    # Grid-based initializations with different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 1.1, 0.9]
    side = int(np.ceil(np.sqrt(N)))

    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4
        configs.append((x, y, r, w0, h0))

    # Hexagonal packing initialization
    for w0 in [1.0, 1.2, 0.85, 1.35]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        spacing = min(w0, h0) / 5.5
        row_y = 0
        col_x = 0
        count = 0
        while count < N and row_y < 10:
            for col in range(10):
                if count >= N:
                    break
                offset = (row_y % 2) * spacing * 0.5
                cx = (col + 0.5) * spacing + offset
                cy = (row_y + 0.5) * spacing * 0.866
                if cx < w0 and cy < h0:
                    x[count] = cx
                    y[count] = cy
                    r[count] = spacing * 0.45
                    count += 1
            row_y += 1
            if count >= N:
                break
        if count == N:
            configs.append((x, y, r, w0, h0))

    # Random initializations
    rng = np.random.default_rng(123)
    for trial in range(15):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            r[i] = rng.uniform(0.04, 0.1)
            x[i] = rng.uniform(r[i], w0 - r[i])
            y[i] = rng.uniform(r[i], h0 - r[i])
        configs.append((x, y, r, w0, h0))

    # --- Phase 1: Force-directed optimization on each config ---
    for ci, (x0, y0, r0, w0, h0) in enumerate(configs):
        if time.time() > deadline - 30:
            break

        time_left = deadline - time.time() - 20
        if time_left < 2:
            break

        # Allocate time per config
        time_per_config = min(8.0, time_left / max(1, len(configs) - ci))

        result = _force_directed_solve(x0, y0, r0, w0, h0, time_per_config,
                                         repel_strength=10.0, wall_strength=50.0,
                                         center_pull=0.01, radius_growth=0.05)
        if result is not None:
            rx, ry, rr, rw, rh, val = result
            if val > best_value:
                best_value = val
                best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

        # Also try with different parameters
        if time.time() < deadline - 30:
            result2 = _force_directed_solve(x0, y0, r0, w0, h0, time_per_config * 0.5,
                                             repel_strength=20.0, wall_strength=80.0,
                                             center_pull=0.02, radius_growth=0.08)
            if result2 is not None:
                rx, ry, rr, rw, rh, val = result2
                if val > best_value:
                    best_value = val
                    best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

    # --- Phase 2: Global expansion on best result ---
    if best_result is not None and time.time() < deadline - 25:
        bx, by, br, bw, bh = best_result
        time_left = deadline - time.time() - 15
        expanded = _global_expansion(bx, by, br, bw, bh, time_left)
        if expanded is not None:
            ex, ey, er, ew, eh, val = expanded
            if val > best_value:
                best_value = val
                best_result = (ex, ey, er, ew, eh)

    # --- Phase 3: Second round of force-directed on best result with tighter params ---
    if best_result is not None and time.time() < deadline - 20:
        bx, by, br, bw, bh = best_result
        time_left = deadline - time.time() - 10
        result = _force_directed_solve(bx, by, br, bw, bh, time_left,
                                         repel_strength=15.0, wall_strength=60.0,
                                         center_pull=0.005, radius_growth=0.03)
        if result is not None:
            rx, ry, rr, rw, rh, val = result
            if val > best_value:
                best_value = val
                best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

        # Another expansion pass
        if best_result is not None and time.time() < deadline - 10:
            bx, by, br, bw, bh = best_result
            time_left = deadline - time.time() - 5
            expanded = _global_expansion(bx, by, br, bw, bh, time_left)
            if expanded is not None:
                ex, ey, er, ew, eh, val = expanded
                if val > best_value:
                    best_value = val
                    best_result = (ex, ey, er, ew, eh)

    # --- Phase 4: Final fine-tuning ---
    if best_result is not None and time.time() < deadline - 8:
        bx, by, br, bw, bh = best_result
        time_left = deadline - time.time() - 3
        result = _force_directed_solve(bx, by, br, bw, bh, time_left,
                                         repel_strength=8.0, wall_strength=40.0,
                                         center_pull=0.003, radius_growth=0.02)
        if result is not None:
            rx, ry, rr, rw, rh, val = result
            if val > best_value:
                best_value = val
                best_result = (rx.copy(), ry.copy(), rr.copy(), rw, rh)

    # --- Fallback ---
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            cr = 0.5 / side - 0.01
            cells.append([cx, cy, cr])
        return np.array(cells)

    # --- Final safety: ensure strict validity ---
    bx, by, br, bw, bh = best_result
    for _ in range(20):
        r_trial = br * (1 - 1e-8)
        if _is_valid(bx, by, r_trial, bw, bh):
            br = r_trial
        else:
            break

    return np.column_stack((bx, by, br))
