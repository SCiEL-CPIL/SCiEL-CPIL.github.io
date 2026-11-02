"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + Multi-precision Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-10):
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


def _max_radius_for_circle(x, y, r, w, h, i):
    """Compute the maximum radius for circle i given all other positions and radii."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _check_pair_valid(nx, ny, x, y, r, w, h, i, tol=1e-10):
    """Check if circle i can be at (nx, ny)."""
    if nx < r[i] - tol or nx > w - r[i] + tol:
        return False
    if ny < r[i] - tol or ny > h - r[i] + tol:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = nx - x[j]
        dy = ny - y[j]
        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
            return False
    return True


def _greedy_expand(x, y, r, w, h, time_budget=None):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 30:
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-12:
                r[i] = r_max
                improved = True
    return x, y, r


def _force_relax(z, max_iter=400):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = 0.004
    min_dt = 0.00005

    iu = np.triu_indices(n, k=1)

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80.0
        fx += right_viol * 80.0
        fy -= bottom_viol * 80.0
        fy += top_viol * 80.0

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-14))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0
        if max_overlap < 1e-10:
            break

        f_pairs = overlap * 20.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / np.maximum(d_pairs, 1e-14)
        uy = dy_pairs / np.maximum(d_pairs, 1e-14)

        np.add.at(fx, iu[0], ux * f_pairs)
        np.add.at(fy, iu[0], uy * f_pairs)
        np.add.at(fx, iu[1], -ux * f_pairs)
        np.add.at(fy, iu[1], -uy * f_pairs)

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.05:
            dt = min(0.01, dt * 1.2)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _epigraph_obj(z_ep, beta=40.0):
    """Epigraph objective: minimize -sum(r) + penalty*t + soft constraints."""
    n = N
    x = z_ep[0:n]
    y = z_ep[n:2*n]
    r = z_ep[2*n:3*n]
    w = z_ep[3*n]
    t = z_ep[3*n+1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r) + 5.0 * max(0.0, t)

    # Pairwise: r_i + r_j - d_ij <= t
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(n, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-14))
    r_sum_pairs = r_sum[iu]
    overlap_vals = r_sum_pairs - d_pairs

    viol = np.maximum(overlap_vals - t, 0.0)
    if np.max(viol) > 1e-12:
        shifted = beta * viol
        shifted = np.clip(shifted, -300, 300)
        m = np.max(shifted)
        lse = m + np.log(np.sum(np.exp(shifted - m)))
        obj += 80.0 * lse / beta
    obj += 40.0 * np.sum(viol * viol)

    # Boundary: violations <= t
    all_viol = np.concatenate([
        np.maximum(r - x - t, 0.0),
        np.maximum(x + r - w - t, 0.0),
        np.maximum(r - y - t, 0.0),
        np.maximum(y + r - h - t, 0.0),
    ])
    if np.max(all_viol) > 1e-12:
        shifted_b = beta * all_viol
        shifted_b = np.clip(shifted_b, -300, 300)
        m_b = np.max(shifted_b)
        lse_b = m_b + np.log(np.sum(np.exp(shifted_b - m_b)))
        obj += 80.0 * lse_b / beta
    obj += 40.0 * np.sum(all_viol * all_viol)

    # Aspect ratio soft penalty
    if w < 0.5:
        obj += 150.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 150.0 * (w - 1.5) ** 2

    return obj


def _local_search(x, y, r, w, h, time_budget, n_trials=12000):
    """Random position perturbations with greedy expansion."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(7)
    best_sum = float(np.sum(r))
    n = N
    iu = np.triu_indices(n, k=1)

    for _ in range(n_trials):
        if time.time() > time_budget:
            break

        if rng.random() < 0.6:
            idxs = [rng.integers(0, n)]
        else:
            k = rng.integers(2, 4)
            idxs = list(rng.choice(n, size=k, replace=False))

        moved = True
        trial_x = x.copy()
        trial_y = y.copy()

        for i in idxs:
            scale = 0.002 + 0.004 * rng.random()
            dx = rng.normal(0, scale)
            dy = rng.normal(0, scale)
            nx = x[i] + dx
            ny = y[i] + dy
            if not _check_pair_valid(nx, ny, x, y, r, w, h, i):
                # Check against other moved circles too
                ok = True
                for j in idxs:
                    if j == i:
                        continue
                    if j < i:  # only check already-set ones
                        ddx = nx - trial_x[j]
                        ddy = ny - trial_y[j]
                        if ddx*ddx + ddy*ddy < (r[i]+r[j])**2 - 1e-9:
                            ok = False
                            break
                if not ok or nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                    moved = False
                    break
            trial_x[i] = nx
            trial_y[i] = ny

        if not moved:
            continue

        # Full pairwise check
        dx_all = trial_x[iu[0]] - trial_x[iu[1]]
        dy_all = trial_y[iu[0]] - trial_y[iu[1]]
        d2_all = dx_all*dx_all + dy_all*dy_all
        rs_all = r[iu[0]] + r[iu[1]]
        if np.any(d2_all < rs_all*rs_all - 1e-9):
            continue

        x = trial_x
        y = trial_y
        x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
        if np.sum(r2) > best_sum + 1e-10:
            best_sum = float(np.sum(r2))
            r = r2
            x = x2
            y = y2

    return x, y, r


def _aspect_search(best_result, time_budget):
    """Try small aspect ratio changes with rescaling."""
    x = best_result[0::3][:N].copy()
    y = best_result[1::3][:N].copy()
    r = best_result[2::3][:N].copy()
    w = best_result[-1]
    h = 2.0 - w

    best_val = float(np.sum(r))
    best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

    for dw in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.003, -0.003,
               0.002, -0.002, 0.001, -0.001, 0.008, -0.008, 0.015, -0.015,
               0.03, -0.03, 0.05, -0.05]:
        if time.time() > time_budget:
            break
        w_t = w + dw
        h_t = 2.0 - w_t
        if w_t < 0.25 or h_t < 0.25:
            continue
        sx = w_t / w
        sy = h_t / h
        x_t = x * sx
        y_t = y * sy
        r_t = r * min(sx, sy)
        if _is_valid(x_t, y_t, r_t, w_t, h_t, tol=1e-8):
            x2, y2, r2 = _greedy_expand(x_t, y_t, r_t, w_t, h_t, time_budget)
            if _is_valid(x2, y2, r2, w_t, h_t):
                val = float(np.sum(r2))
                if val > best_val:
                    best_val = val
                    best_x, best_y, best_r, best_w = x2.copy(), y2.copy(), r2.copy(), w_t

    return best_x, best_y, best_r, best_w


def _high_precision_polish(x, y, r, w, h, time_budget):
    """Iterative expansion + micro-moves + uniform scaling."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    tol = 1e-14
    max_rounds = 600
    rng = np.random.default_rng(42)

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False

        # Expand each circle
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + tol:
                r[i] = r_max
                improved = True

        if improved:
            continue

        # Try micro nudges for constrained circles
        for i in range(n):
            r_max_i = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max_i <= r[i] + 1e-10:
                continue
            # Try fine-grid nudges
            for dx in [0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001]:
                for dy in [0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001]:
                    nx, ny = x[i] + dx, y[i] + dy
                    if _check_pair_valid(nx, ny, x, y, r, w, h, i):
                        x[i], y[i] = nx, ny
                        r_new = _max_radius_for_circle(x, y, r, w, h, i)
                        if r_new > r[i] + tol:
                            r[i] = r_new
                            improved = True
                            break
                if improved:
                    break

        # Random 2-circle moves
        if not improved:
            for _ in range(30):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                dx1, dy1 = rng.normal(0, 0.0003), rng.normal(0, 0.0003)
                dx2, dy2 = rng.normal(0, 0.0003), rng.normal(0, 0.0003)
                nx1, ny1 = x[i]+dx1, y[i]+dy1
                nx2, ny2 = x[j]+dx2, y[j]+dy2
                if not _check_pair_valid(nx1, ny1, x, y, r, w, h, i):
                    continue
                if not _check_pair_valid(nx2, ny2, x, y, r, w, h, j):
                    continue
                # i-j check
                ddx, ddy = nx1-nx2, ny1-ny2
                if ddx*ddx + ddy*ddy < (r[i]+r[j])**2 - tol:
                    continue
                # Cross-checks
                ok = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ddx = nx1 - x[k]
                    ddy = ny1 - y[k]
                    if ddx*ddx + ddy*ddy < (r[i]+r[k])**2 - tol:
                        ok = False
                        break
                    ddx = nx2 - x[k]
                    ddy = ny2 - y[k]
                    if ddx*ddx + ddy*ddy < (r[j]+r[k])**2 - tol:
                        ok = False
                        break
                if not ok:
                    continue
                x[i], y[i] = nx1, ny1
                x[j], y[j] = nx2, ny2
                # Expand all
                for k in range(n):
                    rm = _max_radius_for_circle(x, y, r, w, h, k)
                    if rm > r[k] + tol:
                        r[k] = rm
                        improved = True
                if improved:
                    break
                else:
                    # Revert
                    x[i] -= dx1; y[i] -= dy1
                    x[j] -= dx2; y[j] -= dy2

        # Uniform scaling
        if not improved:
            for f in [1+1e-7, 1+5e-8, 1+2e-8, 1+1e-8, 1+5e-9]:
                r_t = r * f
                if _is_valid(x, y, r_t, w, h, tol=1e-12):
                    r = r_t
                    improved = True
                    break

        if not improved:
            break

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin

    best_value = 0.0
    best_result = None

    def check_validity(z, tol=1e-10):
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
                if dx*dx + dy*dy < (r[i]+r[j])**2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55,
                         1.2, 0.8, 0.9, 1.1, 0.7, 1.4, 0.65, 1.25, 0.78, 1.28]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.44
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.45, 1.55)
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal
        w0 = 1.0; h0 = 1.0
        z = np.zeros(3 * N + 1)
        cols = 5
        sx = w0 / (cols + 0.5)
        sy = h0 / (5 + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.47
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Epigraph SLSQP on diverse starts
    n_starts = min(len(guesses), 14)
    for i in range(n_starts):
        if time.time() > deadline - 50:
            break
        z0 = _force_relax(guesses[i], max_iter=200)
        # Convert to epigraph form
        z_ep = np.zeros(3 * N + 2)
        z_ep[:3*N] = z0[:3*N]
        z_ep[3*N] = z0[-1]
        z_ep[3*N+1] = 0.005

        remaining = deadline - (time.time() - start_time) - 50
        if remaining < 8:
            break
        maxiter = int(min(350, max(80, remaining * 20)))
        try:
            bounds_ep = [(0, 2)] * (2*N) + [(0, 0.8)] * N + [(0.25, 1.75), (0.0, 0.3)]
            res = minimize(_epigraph_obj, z_ep, method='SLSQP',
                           bounds=bounds_ep,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            z_opt = res.x[:3*N+1].copy()
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_rep = z_opt.copy()
                z_rep[2::3][:N] *= 0.95
                valid, val = check_validity(z_rep)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_rep.copy()
        except Exception:
            continue

    # Phase 2: If we have a good result, do more targeted SLSQP from it
    if best_result is not None and time.time() < deadline - 30:
        z0 = _force_relax(best_result, max_iter=100)
        z_ep = np.zeros(3 * N + 2)
        z_ep[:3*N] = z0[:3*N]
        z_ep[3*N] = z0[-1]
        z_ep[3*N+1] = 0.002
        remaining = deadline - (time.time() - start_time) - 30
        if remaining > 10:
            maxiter = int(min(300, max(80, remaining * 15)))
            try:
                bounds_ep = [(0, 2)] * (2*N) + [(0, 0.8)] * N + [(0.25, 1.75), (0.0, 0.3)]
                res = minimize(_epigraph_obj, z_ep, method='SLSQP',
                               bounds=bounds_ep,
                               options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 50})
                z_opt = res.x[:3*N+1].copy()
                valid, val = check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            except Exception:
                pass

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 15:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Aspect ratio search
        if time.time() < deadline - 12:
            x, y, r, w = _aspect_search(best_result, deadline - 12)
            h = 2.0 - w
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # Greedy expand
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 10)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = np.empty(3*N+1)
                best_result[0::3][:N] = x
                best_result[1::3][:N] = y
                best_result[2::3][:N] = r
                best_result[-1] = w

        # Local search
        if time.time() < deadline - 8:
            x, y, r = _local_search(x, y, r, w, h, deadline - 5, n_trials=15000)
            if _is_valid(x, y, r, w, h):
                x, y, r = _greedy_expand(x, y, r, w, h, deadline - 4)
                if _is_valid(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3*N+1)
                        best_result[0::3][:N] = x
                        best_result[1::3][:N] = y
                        best_result[2::3][:N] = r
                        best_result[-1] = w

        # High precision polish
        if time.time() < deadline - 3:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x, y, r = _high_precision_polish(x, y, r, w, h, deadline - 1)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # Final binary search uniform scale
        if time.time() < deadline - 0.5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            lo, hi = 1.0, 1.015
            for _ in range(50):
                if time.time() > deadline - 0.3:
                    break
                mid = (lo + hi) / 2.0
                if _is_valid(x, y, r * mid, w, h, tol=1e-11):
                    lo = mid
                else:
                    hi = mid
            r_final = r * lo
            if _is_valid(x, y, r_final, w, h, tol=1e-10):
                val = float(np.sum(r_final))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r_final
                    best_result[-1] = w
            # Per-circle max
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            for i in range(N):
                rm = _max_radius_for_circle(x, y, r, w, h, i)
                if rm > r[i] + 1e-12:
                    r[i] = rm
            if _is_valid(x, y, r, w, h, tol=1e-10):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety shrink
    for _ in range(30):
        r_t = r * (1 - 5e-11)
        if _is_valid(x, y, r_t, w, h, tol=1e-10):
            r = r_t
        else:
            break

    return np.column_stack((x, y, r))
