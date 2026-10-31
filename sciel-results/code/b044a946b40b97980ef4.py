"""Circle packing in perimeter-4 rectangle: Unified SLSQP with Constraint-Based Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Phase 1: Unified SLSQP optimization with free aspect ratio (Penalty Method) ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w

    n_starts = 12
    penalty_weight = 500.0

    def objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Overlap penalty (smooth)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                dist_req = r[i] + r[j]
                if d < dist_req:
                    obj += penalty_weight * (dist_req - d) ** 2

        # Boundary penalty (smooth)
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

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

    # Generate diverse initial guesses with hexagonal patterns
    def get_initial_guesses():
        guesses = []

        # Hexagonal packing seeds with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.25, 0.9]

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = _hex_seed(w0, h0)
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(n_starts - len(guesses)):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    def _hex_seed(w, h):
        """Create hexagonal packing seed for given rectangle dimensions."""
        z = []
        # Calculate hex spacing
        # For N=21, try different row/column configurations
        configs = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]
        
        best_config = None
        best_radius = 0
        
        for rows, cols in configs:
            if rows * cols < N:
                continue
            # Hex spacing
            dx = w / (cols + 0.5)
            dy = h / (rows + 0.5)
            # Hex vertical offset
            hex_dy = dy * np.sqrt(3) / 2
            r_candidate = min(dx / 2, hex_dy / 2) * 0.95
            if r_candidate > best_radius:
                best_radius = r_candidate
                best_config = (rows, cols)
        
        if best_config is None:
            best_config = (5, 5)
            dx = w / 5.5
            dy = h / 5.5
            best_radius = min(dx, dy) * 0.4
        
        rows, cols = best_config
        dx = w / (cols + 0.5)
        dy = h / (rows + 0.5)
        hex_dy = dy * np.sqrt(3) / 2
        r0 = min(dx / 2, hex_dy / 2) * 0.92
        
        count = 0
        for row in range(rows + 1):
            for col in range(cols + 1):
                if count >= N:
                    break
                # Hex offset for odd rows
                cx = (col + 0.5) * dx + (row % 2) * dx / 2
                cy = (row + 0.5) * hex_dy
                # Clamp to bounds
                cx = np.clip(cx, r0, w - r0)
                cy = np.clip(cy, r0, h - r0)
                z.extend([cx, cy, r0])
                count += 1
        
        # Fill remaining with grid
        if count < N:
            side = int(np.ceil(np.sqrt(N)))
            for i in range(count, N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w
                cy = (row + 0.5) / side * h
                r_grid = min(w / side, h / side) * 0.4
                cx = np.clip(cx, r_grid, w - r_grid)
                cy = np.clip(cy, r_grid, h - r_grid)
                z.extend([cx, cy, r_grid])
        
        z.append(w)
        return np.array(z[:3*N+1])

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=300)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: High-precision polish on best result ---
    if best_result is not None and time.time() < deadline - 20:
        # First: Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # Second: Constraint-based SLSQP refinement with active set
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time))
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Third: Iterative radius growth with local adjustment
        if best_result is not None and time.time() < deadline - 5:
            grown = _iterative_growth(best_result, deadline - (time.time() - start_time))
            if grown is not None:
                valid, val = check_validity(grown)
                if valid and val > best_value:
                    best_value = val
                    best_result = grown

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with small shrink
    for _ in range(50):
        r_trial = r * (1 - 1e-7)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=300):
    """Force-directed relaxation in unified parameter space with adaptive step."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0005

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 25
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 25
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 25
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 25

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 12
                    fy[i] += (dy / d) * overlap * 12
                    fx[j] -= (dx / d) * overlap * 12
                    fy[j] -= (dy / d) * overlap * 12

        # Adaptive step size: reduce if forces are large
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.5:
            dt = max(min_dt, 0.002)
        elif max_force > 0.1:
            dt = max(min_dt, 0.004)
        else:
            dt = 0.005

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(1500, time_budget * 80))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 35
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 35
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 35
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 35

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 18
                    fy[i] += (dy / d) * overlap * 18
                    fx[j] -= (dx / d) * overlap * 18
                    fy[j] -= (dy / d) * overlap * 18

        # Adaptive step
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints with active set detection."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        # Objective is simply -sum(r)
        r = zv[2::3][:n]
        return -np.sum(r)

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7)) # w

    # Detect active constraints to reduce constraint count
    x = z[0::3][:n]
    y = z[1::3][:n]
    r = z[2::3][:n]
    w = z[-1]
    h = 2.0 - w

    # Boundary constraints that are active (within tolerance)
    tol_active = 1e-4
    active_boundary = []
    for i in range(n):
        if x[i] - r[i] < tol_active:
            active_boundary.append((i, 0, 0))  # left wall
        if w - x[i] - r[i] < tol_active:
            active_boundary.append((i, 0, 1))  # right wall
        if y[i] - r[i] < tol_active:
            active_boundary.append((i, 1, 0))  # bottom wall
        if h - y[i] - r[i] < tol_active:
            active_boundary.append((i, 1, 1))  # top wall

    # Overlap constraints that are active
    active_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < (r_sum + tol_active) ** 2:
                active_overlap.append((i, j))

    # If too many active constraints, use all (fallback)
    use_all_constraints = len(active_boundary) > 20 or len(active_overlap) > 50

    constraints = []

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3*idx] - p[3*idx+2]
            else:
                return lambda p: p[-1] - p[3*idx] - p[3*idx+2]
        else:
            if side == 0:
                return lambda p: p[3*idx+1] - p[3*idx+2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3*idx+1] - p[3*idx+2]

    def make_overlap_constraint(idx1, idx2):
        return lambda p: (
            (p[3*idx1] - p[3*idx2])**2 + 
            (p[3*idx1+1] - p[3*idx2+1])**2 - 
            (p[3*idx1+2] + p[3*idx2+2])**2
        )

    if use_all_constraints:
        # Use all constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    else:
        # Use only active constraints
        for (i, axis, side) in active_boundary:
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, axis, side)})
        for (i, j) in active_overlap:
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(300, time_budget * 80))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        # Verify validity
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _iterative_growth(z, time_budget):
    """Iteratively grow radii while adjusting positions to maintain validity."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = np.sum(r)

    # Try uniform growth first
    for factor in [1.001, 1.002, 1.005, 1.01, 1.02]:
        if time.time() > time_budget:
            break
        r_trial = r * factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            best_val = np.sum(r)
        else:
            # Try to fix by local position adjustment
            fixed = _local_fix(x, y, r_trial, w, h)
            if fixed is not None:
                fx, fy, fr = fixed
                if _is_valid(fx, fy, fr, w, h):
                    x, y, r = fx, fy, fr
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    best_val = np.sum(r)

    # Try individual circle growth
    for _ in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_trial = r.copy()
            r_trial[i] *= 1.01
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                best_val = np.sum(r)
                improved = True
            else:
                # Try local fix
                fixed = _local_fix(x, y, r_trial, w, h)
                if fixed is not None:
                    fx, fy, fr = fixed
                    if _is_valid(fx, fy, fr, w, h) and np.sum(fr) > best_val:
                        x, y, r = fx, fy, fr
                        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                        best_val = np.sum(r)
                        improved = True
        if not improved:
            break

    if best_val > np.sum(z[2::3][:n]):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        if _is_valid(best_x, best_y, best_r, w, h):
            return z_new
    return None


def _local_fix(x, y, r, w, h, max_iter=100):
    """Try to fix a slightly invalid packing by local position adjustments."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    n = N

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 50
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 50
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 50
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 50

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        # Check if forces are small enough
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), 1e-8)
        if max_force < 1e-6:
            break

        # Small step
        dt = 0.001
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


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


def _hex_seed(w, h):
    """Create hexagonal packing seed for given rectangle dimensions."""
    z = []
    configs = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]
    
    best_config = None
    best_radius = 0
    
    for rows, cols in configs:
        if rows * cols < N:
            continue
        dx = w / (cols + 0.5)
        dy = h / (rows + 0.5)
        hex_dy = dy * np.sqrt(3) / 2
        r_candidate = min(dx / 2, hex_dy / 2) * 0.95
        if r_candidate > best_radius:
            best_radius = r_candidate
            best_config = (rows, cols)
    
    if best_config is None:
        best_config = (5, 5)
        dx = w / 5.5
        dy = h / 5.5
        best_radius = min(dx, dy) * 0.4
    
    rows, cols = best_config
    dx = w / (cols + 0.5)
    dy = h / (rows + 0.5)
    hex_dy = dy * np.sqrt(3) / 2
    r0 = min(dx / 2, hex_dy / 2) * 0.92
    
    count = 0
    for row in range(rows + 1):
        for col in range(cols + 1):
            if count >= N:
                break
            cx = (col + 0.5) * dx + (row % 2) * dx / 2
            cy = (row + 0.5) * hex_dy
            cx = np.clip(cx, r0, w - r0)
            cy = np.clip(cy, r0, h - r0)
            z.extend([cx, cy, r0])
            count += 1
    
    if count < N:
        side = int(np.ceil(np.sqrt(N)))
        for i in range(count, N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w
            cy = (row + 0.5) / side * h
            r_grid = min(w / side, h / side) * 0.4
            cx = np.clip(cx, r_grid, w - r_grid)
            cy = np.clip(cy, r_grid, h - r_grid)
            z.extend([cx, cy, r_grid])
    
    z.append(w)
    return np.array(z[:3*N+1])
