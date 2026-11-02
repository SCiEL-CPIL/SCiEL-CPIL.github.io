"""Circle packing in perimeter-4 rectangle: grid-cell + force-directed + SLSQP + SLP contact refinement."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21
DEADLINE_MARGIN = 10.0


def _perimeter(w, h):
    return 2.0 * (w + h)


def _radius_sum(circles):
    return float(np.sum(circles[:, 2]))


def _is_valid(circles, tol=1e-9):
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    W = max_x - min_x
    H = max_y - min_y
    if W + H > 2.0 + tol:
        return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            r_sum = rs[i] + rs[j]
            if dist_sq < r_sum * r_sum - tol:
                return False
    return True


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth soft-min penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Smooth pairwise overlap penalty using soft-min
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    r_pairs = r_sum[iu]
    # Soft overlap: smooth approximation of max(r_pairs - d_pairs, 0)
    beta = 20.0
    soft_overlap = np.logaddexp(0.0, beta * (r_pairs - d_pairs)) / beta
    obj += penalty_weight * np.sum(soft_overlap)

    # Smooth boundary penalties
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    for viol in [left_viol, right_viol, bottom_viol, top_viol]:
        soft_viol = np.logaddexp(0.0, beta * viol) / beta
        obj += penalty_weight * np.sum(soft_viol)

    # Aspect ratio soft constraint (smooth)
    w_dev = np.logaddexp(0.0, beta * (0.5 - w)) / beta + np.logaddexp(0.0, beta * (w - 1.5)) / beta
    obj += penalty_weight * w_dev

    return obj


def _force_relax(z, max_iter=300, dt_init=0.003):
    """Force-directed relaxation with vectorized pairwise forces."""
    z = z.copy().astype(float)
    dt = dt_init
    for it in range(max_iter):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Vectorized repulsion between overlapping circles
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - d, 0.0)
        # Only upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        dx_u = dx[iu]
        dy_u = dy[iu]
        d_u = d[iu]
        ov_u = overlap[iu]
        mask = d_u > 1e-12
        safe_d = np.where(mask, d_u, 1.0)
        force = ov_u * 10.0
        fx[iu[0]] += force * dx_u / safe_d
        fy[iu[0]] += force * dy_u / safe_d
        fx[iu[1]] -= force * dx_u / safe_d
        fy[iu[1]] -= force * dy_u / safe_d

        # Boundary forces (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx += 50.0 * (left_viol - right_viol)
        fy += 50.0 * (bottom_viol - top_viol)

        # Apply forces
        z[0::3][:N] += dt * fx
        z[1::3][:N] += dt * fy

        # Clamp
        for i in range(N):
            z[3 * i] = np.clip(z[3 * i], r[i] * 0.5, w - r[i] * 0.5)
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i] * 0.5, h - r[i] * 0.5)

        # Decay step
        dt *= 0.995
        if dt < 1e-6:
            break

    return z


def _grid_cell_layout(w, h, n_cols, n_rows, margin=0.95):
    """
    Grid-based cellular decomposition: partition w x h rectangle into
    n_cols x n_rows cells, place one circle per cell with radius = inradius * margin.
    This guarantees disjoint interiors by construction.
    Returns z vector of shape (3N + 1,).
    """
    total_cells = n_cols * n_rows
    if total_cells < N:
        return None

    z = np.zeros(3 * N + 1)
    cell_w = w / n_cols
    cell_h = h / n_rows

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            # Inradius of the cell is min(cell_w, cell_h)/2; use margin for safety
            r = 0.5 * min(cell_w, cell_h) * margin
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r
            count += 1
    z[-1] = w
    return z


def _expansion_polish(z, time_budget=5.0):
    """
    Try to expand radii while maintaining validity.
    Uses iterative radius scaling with force relaxation.
    """
    z = z.copy().astype(float)
    start = time.time()
    scale = 1.0
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)

    for step in range(50):
        if time.time() - start > time_budget:
            break
        # Try to increase all radii by a small factor
        scale *= 1.02
        z_trial = z.copy()
        z_trial[2::3][:N] = z[2::3][:N] * scale

        # Check if still valid
        if _is_valid_from_z(z_trial):
            z = z_trial
            val = _radius_sum_from_z(z)
            if val > best_val:
                best_val = val
                best_z = z.copy()
        else:
            # Relax to resolve overlaps
            z_relaxed = _force_relax(z_trial, max_iter=100, dt_init=0.002)
            if _is_valid_from_z(z_relaxed):
                z = z_relaxed
                val = _radius_sum_from_z(z)
                if val > best_val:
                    best_val = val
                    best_z = z.copy()
            else:
                break

    return best_z


def _radius_sum_from_z(z):
    r = z[2::3][:N]
    return float(np.sum(r))


def _is_valid_from_z(z, tol=1e-9):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False

    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _slsqp_refine(z0, time_budget=15.0):
    """SLSQP refinement on the unified parameter vector."""
    start = time.time()
    remaining = time_budget - (time.time() - start)
    maxiter = int(min(500, max(100, remaining * 40)))

    try:
        res = minimize(
            _vectorized_objective, z0, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60}
        )
        if res.success or res.fun < _vectorized_objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def _slp_contact_refine(z0, time_budget=10.0):
    """
    SLP (Sequential Linear Programming) refinement using contact graph.
    Identifies active contacts (near-touching pairs and wall contacts),
    then solves a linearized system to adjust positions and radii.
    """
    start = time.time()
    z = z0.copy().astype(float)
    
    for slp_iter in range(30):
        if time.time() - start > time_budget:
            break
        
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        
        # Identify contact pairs (distance close to sum of radii)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        r_sum = r[:, None] + r[None, :]
        
        # Contact threshold: within 2% of touching
        tol = 0.02
        contact_mask = (np.abs(d - r_sum) < tol * np.maximum(r_sum, 0.01)) & (d > 1e-6)
        np.fill_diagonal(contact_mask, False)
        
        # Also include wall contacts
        wall_left = np.abs(x - r) < tol * r
        wall_right = np.abs(x + r - w) < tol * r
        wall_bottom = np.abs(y - r) < tol * r
        wall_top = np.abs(y + r - h) < tol * r
        
        # Build linear system: adjust positions to satisfy contact constraints
        # Variables: dx_i, dy_i, dr_i for each circle, plus dw
        n_vars = 3 * N + 1
        
        # Objective: maximize sum of dr (i.e., minimize -sum(dr))
        c_obj = np.zeros(n_vars)
        c_obj[2::3][:N] = -1.0  # maximize sum of radii
        
        # Constraints: for each contact pair, linearize distance constraint
        # |p_i - p_j| >= r_i + r_j  =>  (p_i - p_j)·(p_i - p_j) >= (r_i + r_j)^2
        # Linearize around current: 2*(p_i-p_j)·(dp_i-dp_j) - 2*(r_i+r_j)*(dr_i+dr_j) >= 0
        
        A_rows = []
        b_rows = []
        
        iu = np.triu_indices(N, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            if contact_mask[i, j]:
                dxij = x[i] - x[j]
                dyij = y[i] - y[j]
                rij = r[i] + r[j]
                if rij > 1e-6:
                    # Linearized constraint: 2*dxij*(dx_i-dx_j) + 2*dyij*(dy_i-dy_j) - 2*rij*(dr_i+dr_j) >= 0
                    row = np.zeros(n_vars)
                    row[3*i] = 2 * dxij
                    row[3*i + 1] = 2 * dyij
                    row[3*j] = -2 * dxij
                    row[3*j + 1] = -2 * dyij
                    row[3*i + 2] = -2 * rij
                    row[3*j + 2] = -2 * rij
                    A_rows.append(row)
                    b_rows.append(0.0)
        
        # Wall contact constraints (linearized)
        for i in range(N):
            if wall_left[i]:
                # x_i >= r_i  =>  dx_i - dr_i >= 0
                row = np.zeros(n_vars)
                row[3*i] = 1.0
                row[3*i + 2] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_right[i]:
                # x_i + r_i <= w  =>  dx_i + dr_i - dw <= 0
                row = np.zeros(n_vars)
                row[3*i] = 1.0
                row[3*i + 2] = 1.0
                row[-1] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_bottom[i]:
                # y_i >= r_i  =>  dy_i - dr_i >= 0
                row = np.zeros(n_vars)
                row[3*i + 1] = 1.0
                row[3*i + 2] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_top[i]:
                # y_i + r_i <= h  =>  dy_i + dr_i + dw <= 0  (since h = 2-w)
                row = np.zeros(n_vars)
                row[3*i + 1] = 1.0
                row[3*i + 2] = 1.0
                row[-1] = 1.0
                A_rows.append(row)
                b_rows.append(0.0)
        
        if not A_rows:
            break
        
        A_arr = np.array(A_rows)
        b_arr = np.array(b_rows)
        
        # Convert to linprog form: minimize c^T x subject to A_ub x <= b_ub, bounds
        # Our constraints are >=, so flip: -A x <= -b
        A_ub = -A_arr
        b_ub = -b_arr
        
        # Bounds: small adjustments
        bounds = [(-0.05, 0.05)] * (3 * N) + [(-0.05, 0.05)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Apply small step (damped)
                step = 0.5
                z[0::3][:N] += step * res.x[0::3][:N]
                z[1::3][:N] += step * res.x[1::3][:N]
                z[2::3][:N] += step * res.x[2::3][:N]
                z[-1] += step * res.x[-1]
                
                # Clamp radii to positive
                z[2::3][:N] = np.maximum(z[2::3][:N], 0.01)
                z[-1] = np.clip(z[-1], 0.3, 1.7)
                
                # Force relax to fix any violations
                z = _force_relax(z, max_iter=50, dt_init=0.001)
                
                if not _is_valid_from_z(z):
                    # Revert if invalid
                    z = z0.copy()
                    break
            else:
                break
        except Exception:
            break
    
    return z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_z = None
    best_val = 0.0

    # --- Phase 1: Grid-cell decompositions with diverse aspect ratios ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.4, 0.7]

    for w0 in aspect_ratios:
        if time.time() > deadline - 35:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try multiple grid decompositions
        for n_cols in range(3, 7):
            for n_rows in range(3, 7):
                if n_cols * n_rows < N:
                    continue
                z = _grid_cell_layout(w0, h0, n_cols, n_rows, margin=0.92)
                if z is None:
                    continue

                # Force relax to settle
                z_relaxed = _force_relax(z, max_iter=200, dt_init=0.003)

                # Check validity
                if not _is_valid_from_z(z_relaxed):
                    continue

                val = _radius_sum_from_z(z_relaxed)
                if val > best_val:
                    best_val = val
                    best_z = z_relaxed.copy()

    # --- Phase 2: SLSQP refinement on top candidates ---
    if best_z is not None:
        if time.time() < deadline - 30:
            refined = _slsqp_refine(best_z, time_budget=15.0)
            if _is_valid_from_z(refined):
                val = _radius_sum_from_z(refined)
                if val > best_val:
                    best_val = val
                    best_z = refined
            else:
                # Try shrink repair
                z_repair = refined.copy()
                z_repair[2::3][:N] *= 0.95
                if _is_valid_from_z(z_repair):
                    val = _radius_sum_from_z(z_repair)
                    if val > best_val:
                        best_val = val
                        best_z = z_repair

    # --- Phase 3: SLP contact graph refinement ---
    if best_z is not None and time.time() < deadline - 20:
        slp_result = _slp_contact_refine(best_z, time_budget=12.0)
        if _is_valid_from_z(slp_result):
            val = _radius_sum_from_z(slp_result)
            if val > best_val:
                best_val = val
                best_z = slp_result.copy()

    # --- Phase 4: Expansion polish ---
    if best_z is not None and time.time() < deadline - 12:
        expanded = _expansion_polish(best_z, time_budget=7.0)
        if _is_valid_from_z(expanded):
            val = _radius_sum_from_z(expanded)
            if val > best_val:
                best_val = val
                best_z = expanded

    # --- Phase 5: Second SLP pass after expansion ---
    if best_z is not None and time.time() < deadline - 8:
        slp_result2 = _slp_contact_refine(best_z, time_budget=5.0)
        if _is_valid_from_z(slp_result2):
            val = _radius_sum_from_z(slp_result2)
            if val > best_val:
                best_val = val
                best_z = slp_result2.copy()

    # --- Phase 6: Final SLSQP pass ---
    if best_z is not None and time.time() < deadline - 5:
        refined2 = _slsqp_refine(best_z, time_budget=4.0)
        if _is_valid_from_z(refined2):
            val = _radius_sum_from_z(refined2)
            if val > best_val:
                best_val = val
                best_z = refined2
        else:
            z_repair = refined2.copy()
            z_repair[2::3][:N] *= 0.96
            if _is_valid_from_z(z_repair):
                val = _radius_sum_from_z(z_repair)
                if val > best_val:
                    best_val = val
                    best_z = z_repair

    # --- Phase 7: Final force polish ---
    if best_z is not None and time.time() < deadline - 2:
        polished = _force_relax(best_z, max_iter=100, dt_init=0.001)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # Fallback
    if best_z is None:
        # Simple 5x5 grid in unit square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])

    # Format output
    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]

    # Safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        circles_trial = np.column_stack((x, y, r_trial))
        if _is_valid(circles_trial):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
