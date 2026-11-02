"""Circle packing in perimeter-4 rectangle: Cellular Init + Delaunay Refinement + SLSQP + Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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


def _cellular_init(w, h, rows, cols, rng):
    """Initialize circles in a grid of cells with jitter."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    total_cells = rows * cols
    while total_cells < N:
        cols += 1
        total_cells = rows * cols

    for i in range(N):
        row = i // cols
        col = i % cols

        cell_w = w / cols
        cell_h = h / rows
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        max_r = min(cell_w, cell_h) / 2.0
        r0 = max_r * 0.92

        jitter_x = rng.uniform(-0.02, 0.02) * cell_w
        jitter_y = rng.uniform(-0.02, 0.02) * cell_h
        cx = np.clip(cx + jitter_x, r0, w - r0)
        cy = np.clip(cy + jitter_y, r0, h - r0)

        x[i] = cx
        y[i] = cy
        r[i] = r0

    return x, y, r


def _force_relax(x, y, r, w, h, max_iter=1500, dt_init=0.004):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    rng = np.random.default_rng(99)

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 60
        fx += right_viol * 60
        fy -= bottom_viol * 60
        fy += top_viol * 60

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

        if np.max(overlap) < 1e-9:
            break

        f_pairs = overlap * 14.0
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
            dt = min(0.006, dt * 1.1)

        # Periodic jitter to escape local minima
        if it % 80 == 40 and it > 0:
            x += rng.normal(0, 0.0008, n)
            y += rng.normal(0, 0.0008, n)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _compute_inradii(x, y, w, h):
    """Compute maximum inscribed circle radius for each point using Delaunay triangulation."""
    n = N
    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
        [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
        [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
    ])
    all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])

    try:
        tri = Delaunay(all_pts)
    except Exception:
        return np.full(n, min(w, h) / 6.0)

    inradii = np.zeros(n)
    for i in range(n):
        min_dist = min(x[i], w - x[i], y[i], h - y[i])

        for simplex in tri.simplices:
            if i in simplex:
                pts = all_pts[simplex]
                ax, ay = pts[0]
                bx, by = pts[1]
                cx, cy = pts[2]

                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

                dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                min_dist = min(min_dist, dist)

        inradii[i] = min_dist

    return inradii


def _delaunay_refine(x, y, r, w, h, max_iter=150, dt=0.003):
    """Refine positions toward Voronoi cell centers to increase inradii."""
    n = N
    x = x.copy()
    y = y.copy()

    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
    ])

    for it in range(max_iter):
        all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])

        try:
            tri = Delaunay(all_pts)
        except Exception:
            break

        new_x = x.copy()
        new_y = y.copy()

        for i in range(n):
            cx_sum = 0.0
            cy_sum = 0.0
            weight_sum = 0.0

            for simplex in tri.simplices:
                if i in simplex:
                    pts = all_pts[simplex]
                    ax, ay = pts[0]
                    bx, by = pts[1]
                    cx, cy = pts[2]

                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                    uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

                    dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                    weight = 1.0 / max(dist, 0.01)
                    cx_sum += ux * weight
                    cy_sum += uy * weight
                    weight_sum += weight

            if weight_sum > 0:
                new_x[i] = cx_sum / weight_sum
                new_y[i] = cy_sum / weight_sum

        x = x + dt * (new_x - x)
        y = y + dt * (new_y - y)
        x = np.clip(x, 0.01, w - 0.01)
        y = np.clip(y, 0.01, h - 0.01)

    # Recompute radii and relax
    r = _compute_inradii(x, y, w, h)
    r = np.maximum(r, 0.005)
    x, y = _force_relax(x, y, r, w, h, max_iter=400)

    if _is_valid(x, y, r, w, h):
        return x, y, r
    else:
        r *= 0.95
        if _is_valid(x, y, r, w, h):
            return x, y, r
    return None


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP with explicit inequality constraints."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    remaining = time_budget - time.time()
    if remaining < 5:
        return None

    bounds = [(0.005, 2.0)] * (2 * n) + [(0.005, 1.0)] * n + [(0.3, 1.7)]

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })

    def objective(zv):
        return -np.sum(zv[2::3][:n])

    maxiter = int(min(500, max(100, remaining * 25)))

    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
        else:
            r_repair = r_opt * 0.99
            if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                return x_opt, y_opt, r_repair, w_opt, h_opt
    except Exception:
        pass
    return None


def _polish_positions(x, y, r, w, h, time_budget):
    """High-precision force-directed polish followed by radius growth."""
    n = N
    x = x.copy()
    y = y.copy()

    max_iter = int(min(3000, time_budget * 150))
    dt = 0.0008
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

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
            f_pairs = overlap * 20.0
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

    # Grow radii iteratively
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _expand_individual(x, y, r, w, h, time_budget):
    """Expand each circle to its maximum possible radius at current position."""
    n = N
    improved = False
    for _ in range(50):
        if time.time() > time_budget:
            break
        changed = False
        for i in range(n):
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

            if r_max > r[i] + 1e-7:
                r[i] = r_max
                changed = True
        if not changed:
            break
        improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def _try_layout(w0, h0, rows, cols, rng, deadline):
    """Try a single layout: cellular init -> relax -> Delaunay refine -> return best."""
    x, y, r = _cellular_init(w0, h0, rows, cols, rng)
    x, y = _force_relax(x, y, r, w0, h0, max_iter=600)

    if not _is_valid(x, y, r, w0, h0):
        r *= 0.95
        if not _is_valid(x, y, r, w0, h0):
            return None

    val = float(np.sum(r))

    # Delaunay refinement
    if time.time() < deadline - 20:
        refined = _delaunay_refine(x, y, r, w0, h0, max_iter=120)
        if refined is not None:
            rx, ry, rr = refined
            rval = float(np.sum(rr))
            if rval > val:
                x, y, r = rx, ry, rr
                val = rval

    return (x.copy(), y.copy(), r.copy(), w0, h0, val)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    rng = np.random.default_rng(42)

    # Phase 1: Try multiple aspect ratios and grid layouts
    aspect_configs = [
        (1.0, 4, 6), (0.85, 5, 5), (1.2, 4, 6), (0.75, 5, 5),
        (1.35, 4, 6), (0.65, 5, 5), (1.5, 4, 6), (0.55, 5, 5),
        (1.1, 3, 7), (0.9, 4, 6), (1.4, 3, 7), (0.8, 4, 6),
    ]

    for w0, rows, cols in aspect_configs:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        result = _try_layout(w0, h0, rows, cols, rng, deadline)
        if result is not None:
            x, y, r, w, h, val = result
            if val > best_value:
                best_value = val
                best_result = (x, y, r, w, h)

    # Phase 2: SLSQP optimization on best
    if best_result is not None and time.time() < deadline - 35:
        x_b, y_b, r_b, w_b, h_b = best_result
        opt_result = _slsqp_optimize(x_b, y_b, r_b, w_b, h_b, deadline)
        if opt_result is not None:
            x_o, y_o, r_o, w_o, h_o = opt_result
            val = float(np.sum(r_o))
            if val > best_value:
                best_value = val
                best_result = (x_o, y_o, r_o, w_o, h_o)

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 15:
        x_b, y_b, r_b, w_b, h_b = best_result
        remaining = deadline - 5
        polished = _polish_positions(x_b, y_b, r_b, w_b, h_b, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_b, h_b)

    # Phase 4: Final individual expansion
    if best_result is not None and time.time() < deadline - 3:
        x_b, y_b, r_b, w_b, h_b = best_result
        expanded = _expand_individual(x_b, y_b, r_b, w_b, h_b, deadline - 1)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_b, h_b)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
