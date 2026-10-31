"""Circle packing in perimeter-4 rectangle: Sequential hard-core placement + aspect-ratio scale search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(x, y, r, W, H, tol=1e-9):
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_r_for_circle(x, y, r, i, W, H):
    """Max radius circle i can have at (x[i], y[i]) given other circles."""
    mr = min(x[i], W - x[i], y[i], H - y[i])
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        allowed = d - r[j]
        if allowed < mr:
            mr = allowed
    return max(0.0, mr)


def _place_circle(idx, x, y, r, W, H, deadline):
    """
    Place circle idx at position that maximizes its clearance (hard-core potential).
    Uses Nelder-Mead on a smooth objective: min over all constraints of (clearance).
    """
    if time.time() > deadline:
        return False

    # Objective: maximize the minimum clearance for circle idx
    # clearance to wall = min(x, W-x, y, H-y)
    # clearance to circle j = dist(i,j) - r[j]
    def neg_clearance(pos):
        cx, cy = pos
        if cx < 0 or cx > W or cy < 0 or cy > H:
            return 100.0
        min_cl = min(cx, W - cx, cy, H - cy)
        for j in range(N):
            if j == idx:
                continue
            dx = cx - x[j]
            dy = cy - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            cl = d - r[j]
            if cl < min_cl:
                min_cl = cl
        return -min_cl

    # Multi-start Nelder-Mead
    best_pos = None
    best_val = 1e18

    # Try several initial guesses
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline:
            break
        if s == 0:
            # Center
            x0 = W / 2.0
            y0 = H / 2.0
        elif s == 1:
            x0 = W * 0.25
            y0 = H * 0.25
        elif s == 2:
            x0 = W * 0.75
            y0 = H * 0.25
        elif s == 3:
            x0 = W * 0.25
            y0 = H * 0.75
        elif s == 4:
            x0 = W * 0.75
            y0 = H * 0.75
        else:
            x0 = W * (0.2 + 0.6 * (s / n_starts))
            y0 = H * (0.2 + 0.6 * ((s * 7) % n_starts) / n_starts)

        res = minimize(neg_clearance, [x0, y0], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best_pos = res.x

    if best_pos is None:
        return False

    # Clamp to valid region (radius 0 for placement)
    cx = float(np.clip(best_pos[0], 0.0, W))
    cy = float(np.clip(best_pos[1], 0.0, H))
    x[idx] = cx
    y[idx] = cy
    r[idx] = _max_r_for_circle(x, y, r, idx, W, H)
    return r[idx] > 1e-6


def _sequential_place(W, H, deadline, order='corners_first'):
    """
    Place N circles sequentially. Order: corners first (largest potential radius),
    then edges, then center.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    # Define placement order: prioritize positions with high "free space"
    # Corners allow largest circles (constrained by 2 walls), then edges, then interior
    # We'll use a heuristic: place in order of decreasing expected max radius
    # For a rectangle, corners > edge-midpoints > center
    # Simple approach: place in a grid-like order from corners inward

    # Create a priority list of (row, col) in a virtual grid
    # For N=21, use a 5x5 grid (25 cells), pick 21
    # Order: corners first, then edges, then interior
    grid_positions = []
    # 5x5 grid indices (0-4, 0-4)
    for row in range(5):
        for col in range(5):
            # Distance from center (2,2) - farther = corner = higher priority
            d_from_center = abs(row - 2) + abs(col - 2)
            # Also prefer corners: min distance to any corner
            corner_d = min(abs(row) + abs(col),
                           abs(row - 4) + abs(col),
                           abs(row) + abs(col - 4),
                           abs(row - 4) + abs(col - 4))
            # Higher priority = smaller corner_d, then smaller d_from_center
            priority = corner_d * 10 + d_from_center
            grid_positions.append((priority, row, col))

    grid_positions.sort(key=lambda t: t[0])

    # Take first N positions
    selected = grid_positions[:N]

    # Map grid to rectangle coordinates
    for i, (prio, row, col) in enumerate(selected):
        if time.time() > deadline:
            break
        # Grid (0..4) maps to rectangle with margin
        cx = (col + 0.5) * W / 5.0
        cy = (row + 0.5) * H / 5.0
        # Initial guess
        x[i] = cx
        y[i] = cy
        r[i] = 0.0  # Will be computed by _place_circle

        # Place this circle
        _place_circle(i, x, y, r, W, H, deadline)

    # Fill any remaining (shouldn't happen if N <= 25)
    for i in range(len(selected), N):
        if time.time() > deadline:
            break
        x[i] = W / 2.0 + (i - N // 2) * 0.05
        y[i] = H / 2.0 + (i - N // 2) * 0.05
        r[i] = 0.03
        _place_circle(i, x, y, r, W, H, deadline)

    return x, y, r


def _scale_search(x0, y0, r0, deadline, n_widths=25):
    """
    Try different aspect ratios (W + H = 2). For each, scale positions and
    check validity. Return best (x, y, r, W, H).
    """
    best_val = 0.0
    best = None

    # Determine initial aspect from r0's bounding box
    if np.max(r0) > 0:
        # Estimate natural aspect from the packing
        x_range = np.max(x0) - np.min(x0)
        y_range = np.max(y0) - np.min(y0)
        if x_range > 0 and y_range > 0:
            natural_aspect = x_range / y_range
            # W/H = aspect, W+H=2 => W = 2*aspect/(1+aspect)
            W_nat = 2.0 * natural_aspect / (1.0 + natural_aspect)
        else:
            W_nat = 1.0
    else:
        W_nat = 1.0

    # Search around natural aspect
    W_vals = np.linspace(max(0.3, W_nat - 0.4), min(1.7, W_nat + 0.4), n_widths)
    # Ensure we include some extremes
    W_vals = np.unique(np.concatenate([W_vals, [0.4, 0.6, 1.0, 1.4, 1.6]]))
    W_vals = W_vals[(W_vals >= 0.3) & (W_vals <= 1.7)]

    for W in W_vals:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H < 0.3:
            continue

        # Scale positions to fit in [0, W] x [0, H]
        # First, normalize x0, y0 to [0,1] then scale
        x_min, x_max = np.min(x0), np.max(x0)
        y_min, y_max = np.min(y0), np.max(y0)
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        x_s = (x0 - x_min) / x_range * W
        y_s = (y0 - y_min) / y_range * H

        # Clamp to valid region
        x_s = np.clip(x_s, r0, W - r0)
        y_s = np.clip(y_s, r0, H - r0)

        if not _is_valid(x_s, y_s, r0, W, H):
            continue

        val = float(np.sum(r0))
        if val > best_val:
            best_val = val
            best = (x_s.copy(), y_s.copy(), r0.copy(), W, H)

    return best


def _slsqp_polish(x, y, r, W, H, deadline):
    """
    SLSQP polish: maximize sum(r) subject to non-overlap and wall constraints.
    """
    if time.time() > deadline:
        return x, y, r

    n = N
    # Variables: x (n), y (n), r (n) = 3n variables
    def objective(v):
        return -np.sum(v[2*n:3*n])  # Maximize sum of radii

    constraints = []

    # Wall constraints: x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H
    # Non-overlap: dist(i,j) >= r_i + r_j

    # Use inequality constraints: g(v) >= 0
    # x_i - r_i >= 0
    for i in range(n):
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[i] - v[2*n + i]})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: W - v[i] - v[2*n + i]})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[n + i] - v[2*n + i]})
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: H - v[n + i] - v[2*n + i]})

    # Non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq',
                                'fun': lambda v, i=i, j=j:
                                (v[i] - v[j])**2 + (v[n+i] - v[n+j])**2 - (v[2*n+i] + v[2*n+j])**2})

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W))  # x_i
        bounds.append((1e-6, H))  # y_i
        bounds.append((1e-6, min(W, H) / 2))  # r_i

    v0 = np.concatenate([x, y, r])

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success and _is_valid(res.x[:n], res.x[n:2*n], res.x[2*n:3*n], W, H):
            return res.x[:n], res.x[n:2*n], res.x[2*n:3*n]
    except Exception:
        pass

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (evaluator kills at 150s)

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    best_H = 1.0

    rng = np.random.default_rng(42)

    # Phase 1: Sequential placement with different initial aspect ratios
    # The sequential placement itself determines the packing structure
    aspect_trials = [1.0, 0.8, 1.25, 0.7, 1.4, 0.65, 1.5, 0.6, 1.3, 0.75,
                     0.5, 1.6, 0.55, 1.45, 0.9]

    for W0 in aspect_trials:
        if time.time() > deadline - 30:
            break
        H0 = 2.0 - W0
        if H0 < 0.3:
            continue

        t_budget = deadline - (time.time() - start_time) - 20
        if t_budget < 5:
            break

        # Sequential placement
        x1, y1, r1 = _sequential_place(W0, H0, start_time + t_budget)

        if not _is_valid(x1, y1, r1, W0, H0):
            continue

        # Scale search: try different aspect ratios for this packing
        t_budget2 = deadline - (time.time() - start_time) - 15
        if t_budget2 > 5:
            scaled = _scale_search(x1, y1, r1, start_time + t_budget2, n_widths=20)
            if scaled is not None:
                x2, y2, r2, W2, H2 = scaled
            else:
                x2, y2, r2, W2, H2 = x1, y1, r1, W0, H0
        else:
            x2, y2, r2, W2, H2 = x1, y1, r1, W0, H0

        if _is_valid(x2, y2, r2, W2, H2):
            val = float(np.sum(r2))
            if val > best_value:
                best_value = val
                best_x = x2.copy()
                best_y = y2.copy()
                best_r = r2.copy()
                best_W = W2
                best_H = H2

    # Phase 2: SLSQP polish on best result
    if best_x is not None and time.time() < deadline - 20:
        t_budget = deadline - (time.time() - start_time) - 10
        if t_budget > 10:
            x3, y3, r3 = _slsqp_polish(best_x, best_y, best_r, best_W, best_H,
                                       start_time + t_budget)
            if _is_valid(x3, y3, r3, best_W, best_H):
                val = float(np.sum(r3))
                if val > best_value:
                    best_value = val
                    best_x = x3.copy()
                    best_y = y3.copy()
                    best_r = r3.copy()

    # Phase 3: Final scale search on polished result
    if best_x is not None and time.time() < deadline - 10:
        t_budget = deadline - (time.time() - start_time) - 5
        if t_budget > 5:
            scaled = _scale_search(best_x, best_y, best_r, start_time + t_budget, n_widths=15)
            if scaled is not None:
                x4, y4, r4, W4, H4 = scaled
                if _is_valid(x4, y4, r4, W4, H4):
                    val = float(np.sum(r4))
                    if val > best_value:
                        best_value = val
                        best_x = x4.copy()
                        best_y = y4.copy()
                        best_r = r4.copy()
                        best_W = W4
                        best_H = H4

    # Fallback
    if best_x is None:
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    for _ in range(20):
        r_trial = best_r * (1 - 1e-8)
        if _is_valid(best_x, best_y, r_trial, best_W, best_H):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))
