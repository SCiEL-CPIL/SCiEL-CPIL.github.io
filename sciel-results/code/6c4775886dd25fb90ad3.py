"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Vectorized smooth objective (softplus for non-overlap, squared for boundary)
 - Variable aspect ratio optimization
 - L-BFGS-B with multiple starts and aspect ratios
 - Vectorized simulated annealing refinement
 - Force-directed relaxation
 - Final high-precision polish
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus(x, beta=10.0):
    """Numerically stable softplus, vectorized."""
    result = np.empty_like(x)
    mask = x > 1.0 / beta
    result[mask] = x[mask]
    result[~mask] = np.log1p(np.exp(beta * x[~mask])) / beta
    return result


def _objective(params, w, h, beta=10.0):
    """
    Objective: minimize penalty - sum(r).
    params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]

    if np.any(rs <= 0):
        return 1e10 + np.sum(np.minimum(rs, 0) ** 2)

    # Vectorized overlap penalty
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[idx_i] + rs[idx_j]
    overlap_vals = min_dist - dist
    overlap_pen = np.sum(_softplus(overlap_vals, beta))

    # Boundary penalty
    pen = 0.0
    viol = np.maximum(0, rs - xs) + np.maximum(0, xs + rs - w)
    viol += np.maximum(0, rs - ys) + np.maximum(0, ys + rs - h)
    boundary_pen = np.sum(viol ** 2)

    return overlap_pen + boundary_pen - np.sum(rs)


def _check_and_update(params, w, h, best_state):
    """Check validity and update best if improved."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]

    if np.any(rs < 1e-9):
        return
    if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
        return
    if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
        return

    # Vectorized overlap check
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist_sq = dx * dx + dy * dy
    min_dist = rs[idx_i] + rs[idx_j]
    if np.any(dist_sq < (min_dist - 1e-7) ** 2):
        return

    val = np.sum(rs)
    if val > best_state['value']:
        best_state['value'] = val
        best_state['result'] = np.zeros((N, 3))
        best_state['result'][:, 0] = xs
        best_state['result'][:, 1] = ys
        best_state['result'][:, 2] = rs


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result


def _hex_initial(w, h):
    """Create hexagonal lattice initial arrangement."""
    x0 = np.zeros(3 * N)
    r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
    r_current = r_est
    positions = []

    for attempt in range(20):
        positions = []
        y = r_current
        row_idx = 0
        while y <= h - r_current + 1e-9:
            if row_idx % 2 == 0:
                x_start = r_current
            else:
                x_start = r_current + r_current
            x = x_start
            while x <= w - r_current + 1e-9:
                positions.append((x, y, r_current))
                x += 2.0 * r_current
            y += np.sqrt(3) * r_current
            row_idx += 1
        if len(positions) >= N:
            break
        else:
            r_current *= 0.95

    if len(positions) < N:
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        positions = []
        for i in range(N):
            col = i % a
            row = i // a
            positions.append(((col + 0.5) * w / a, (row + 0.5) * h / b, min(w / a, h / b) * 0.45))

    for i in range(N):
        if i < len(positions):
            x0[3 * i] = positions[i][0]
            x0[3 * i + 1] = positions[i][1]
            x0[3 * i + 2] = positions[i][2]
        else:
            x0[3 * i] = w / 2
            x0[3 * i + 1] = h / 2
            x0[3 * i + 2] = r_est * 0.5

    return x0


def _grid_initial(w, h):
    """Create grid arrangement of 21 circles."""
    x0 = np.zeros(3 * N)
    best_positions = None
    best_mean_r = 0

    for rows in range(2, 10):
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            continue
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                gx = (j + 0.5) / cols * w
                gy = (i + 0.5) / rows * h
                r = 0.5 * min(w / cols, h / rows) * 0.9
                positions.append((gx, gy, r))
        if len(positions) >= N:
            mean_r = np.mean([p[2] for p in positions])
            if mean_r > best_mean_r:
                best_mean_r = mean_r
                best_positions = positions

    if best_positions is None:
        a, b = 7, 3
        best_positions = []
        for i in range(a):
            for j in range(b):
                if len(best_positions) >= N:
                    break
                best_positions.append(((i + 0.5) * w / a, (j + 0.5) * h / b, min(w / a, h / b) * 0.45))

    for i in range(N):
        x0[3 * i] = best_positions[i][0]
        x0[3 * i + 1] = best_positions[i][1]
        x0[3 * i + 2] = best_positions[i][2]

    return x0


def _random_initial(w, h, rng, r_est):
    """Create random non-overlapping initial state."""
    x0 = np.zeros(3 * N)
    positions = []
    max_attempts = 500

    for i in range(N):
        placed = False
        for attempt in range(max_attempts):
            r = r_est * rng.uniform(0.7, 1.3)
            x = rng.uniform(r, w - r)
            y = rng.uniform(r, h - r)
            ok = True
            for (px, py, pr) in positions:
                if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2:
                    ok = False
                    break
            if ok:
                positions.append((x, y, r))
                placed = True
                break
        if not placed:
            r = r_est * 0.5
            x = rng.uniform(r, w - r)
            y = rng.uniform(r, h - r)
            positions.append((x, y, r))

    for i in range(N):
        x0[3 * i] = positions[i][0]
        x0[3 * i + 1] = positions[i][1]
        x0[3 * i + 2] = positions[i][2]

    return x0


def _force_directed_relaxation(w, h, x0, max_iters=300, deadline=None):
    """Force-directed relaxation to improve contact graph."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()

    step_size = 0.005
    decay = 0.9995

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = xs[idx_i] - xs[idx_j]
        dy = ys[idx_i] - ys[idx_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist[dist < 1e-10] = 1e-10
        min_dist = rs[idx_i] + rs[idx_j]

        overlap = np.maximum(0, min_dist - dist)
        force_mag = overlap * 20.0

        for k in range(len(idx_i)):
            i, j = idx_i[k], idx_j[k]
            if force_mag[k] > 0:
                fx[i] += force_mag[k] * dx[k] / dist[k]
                fy[i] += force_mag[k] * dy[k] / dist[k]
                fx[j] -= force_mag[k] * dx[k] / dist[k]
                fy[j] -= force_mag[k] * dy[k] / dist[k]

        # Boundary forces
        for i in range(N):
            if xs[i] < rs[i]:
                fx[i] += (rs[i] - xs[i]) * 50.0
            if xs[i] > w - rs[i]:
                fx[i] -= (xs[i] - (w - rs[i])) * 50.0
            if ys[i] < rs[i]:
                fy[i] += (rs[i] - ys[i]) * 50.0
            if ys[i] > h - rs[i]:
                fy[i] -= (ys[i] - (h - rs[i])) * 50.0

        r_max = min(w, h) / 2.0
        for i in range(N):
            xs[i] += step_size * fx[i]
            ys[i] += step_size * fy[i]
            xs[i] = np.clip(xs[i], rs[i], w - rs[i])
            ys[i] = np.clip(ys[i], rs[i], h - rs[i])
            rs[i] = np.clip(rs[i], 1e-4, r_max)

        step_size *= decay

    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def _sa_optimize(w, h, x0, max_iters=2000, seed=42, deadline=None):
    """Vectorized simulated annealing for fixed w, h."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()

    r_max = min(w, h) / 2.0
    T = 0.03
    cooling = 0.9995
    rng = np.random.default_rng(seed)

    def current_cost():
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = xs[idx_i] - xs[idx_j]
        dy = ys[idx_i] - ys[idx_j]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = rs[idx_i] + rs[idx_j]
        overlap = np.maximum(0, min_dist - dist)
        return np.sum(overlap) - np.sum(rs)

    curr = current_cost()

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        idx = rng.integers(0, N)
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]

        scale = T * 0.3
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)
        rs[idx] += rng.normal(0, scale * 0.3)

        xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
        ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
        rs[idx] = np.clip(rs[idx], 1e-4, r_max)

        new_c = current_cost()
        delta = new_c - curr

        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            curr = new_c
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r

        T *= cooling

    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_state = {'value': -np.inf, 'result': None}
    rng = np.random.default_rng(42)

    # Try multiple aspect ratios
    t_values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    for t in t_values:
        if time.time() > deadline - 15.0:
            break
        w = 2.0 * t
        h = 2.0 * (1.0 - t)

        r_est = np.sqrt(0.9069 * w * h / (N * np.pi))
        r_max = min(w, h) / 2.0

        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))

        # Multiple initializations
        inits = [
            _hex_initial(w, h),
            _grid_initial(w, h),
            _random_initial(w, h, rng, r_est),
        ]

        best_local_x = None
        best_local_val = -np.inf

        for x0 in inits:
            if time.time() > deadline - 10.0:
                break

            try:
                # L-BFGS-B pass 1
                res = minimize(lambda p: _objective(p, w, h, beta=5.0), x0,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-8})
                _check_and_update(res.x, w, h, best_state)
                val = np.sum(res.x[2::3])
                if val > best_local_val:
                    best_local_val = val
                    best_local_x = res.x.copy()

                # L-BFGS-B pass 2 with higher beta
                if time.time() < deadline - 8.0:
                    res2 = minimize(lambda p: _objective(p, w, h, beta=20.0), res.x,
                                    method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                    _check_and_update(res2.x, w, h, best_state)
                    val2 = np.sum(res2.x[2::3])
                    if val2 > best_local_val:
                        best_local_val = val2
                        best_local_x = res2.x.copy()

            except Exception:
                pass

        # Force-directed on best local
        if best_local_x is not None and time.time() < deadline - 10.0:
            x_fd = _force_directed_relaxation(w, h, best_local_x, max_iters=200, deadline=deadline - 8.0)
            _check_and_update(x_fd, w, h, best_state)

            if time.time() < deadline - 6.0:
                try:
                    res_fd = minimize(lambda p: _objective(p, w, h, beta=30.0), x_fd,
                                      method='L-BFGS-B', bounds=bounds,
                                      options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10})
                    _check_and_update(res_fd.x, w, h, best_state)
                except Exception:
                    pass

        # SA refinement
        if best_local_x is not None and time.time() < deadline - 8.0:
            x_sa = _sa_optimize(w, h, best_local_x, max_iters=1000, seed=42, deadline=deadline - 5.0)
            _check_and_update(x_sa, w, h, best_state)

            if time.time() < deadline - 4.0:
                x_sa2 = _sa_optimize(w, h, best_local_x, max_iters=800, seed=123, deadline=deadline - 3.0)
                _check_and_update(x_sa2, w, h, best_state)

    # Final polish
    if best_state['result'] is not None and time.time() < deadline - 3.0:
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]

        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y

        if bw > 0.01 and bh > 0.01:
            x_polish = np.zeros(3 * N)
            x_polish[0::3] = xs - min_x
            x_polish[1::3] = ys - min_y
            x_polish[2::3] = rs

            bounds_p = []
            r_max_p = min(bw, bh) / 2.0
            for i in range(N):
                bounds_p.append((0, bw))
                bounds_p.append((0, bh))
                bounds_p.append((1e-4, r_max_p))

            try:
                x_fd_p = _force_directed_relaxation(bw, bh, x_polish, max_iters=200, deadline=deadline - 1.0)
                _check_and_update(x_fd_p, bw, bh, best_state)

                res_p = minimize(lambda p: _objective(p, bw, bh, beta=50.0), x_fd_p,
                                 method='L-BFGS-B', bounds=bounds_p,
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res_p.x, bw, bh, best_state)
            except Exception:
                pass

    # Fallback
    if best_state['result'] is None or best_state['value'] <= 0:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r
        _scale_to_perimeter(best_result)
        return best_result

    result = best_state['result'].copy()
    _scale_to_perimeter(result)
    return result
