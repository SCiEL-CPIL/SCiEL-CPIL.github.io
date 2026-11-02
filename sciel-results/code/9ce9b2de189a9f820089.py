"""Circle packing in perimeter-4 rectangle: SLSQP + Epigraph + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _objective_vectorized(z, penalty_weight=2000.0):
    """Vectorized objective with smooth penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r < 1e-4):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary
    left_v = np.maximum(r - x, 0.0)
    right_v = np.maximum(x + r - w, 0.0)
    bot_v = np.maximum(r - y, 0.0)
    top_v = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(left_v**2) + np.sum(right_v**2) + np.sum(bot_v**2) + np.sum(top_v**2))

    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _epigraph_objective(z, beta=20.0):
    """
    Epigraph-style objective: minimize -sum(r) + smooth_max(overlap, boundary_violations).
    z = [x_1..x_n, y_1..y_n, r_1..r_n, t, w]  (t = smooth max violation variable)
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    t = z[3*N]
    w = z[3*N+1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15:
        return 1e8

    # Collect all violations
    violations = []

    # Pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    violations.append(r_sum[iu] - d_pairs)

    # Boundary
    violations.append(r - x)
    violations.append(x + r - w)
    violations.append(r - y)
    violations.append(y + r - h)

    v = np.concatenate(violations)
    v_max = np.max(v)

    # Smooth max
    if v_max < -1.0/beta:
        smax = 0.0
    else:
        smax = np.log(np.sum(np.exp(beta * (v - v_max)))) / beta + v_max

    # Objective: -sum(r) + lambda * t + penalty for t being too small
    lam = 50.0
    obj = -np.sum(r) + lam * t + 1000.0 * np.maximum(smax - t, 0.0) ** 2

    return obj


def _epigraph_obj_grad(z, beta=20.0):
    """Epigraph objective with gradient for L-BFGS-B."""
    x = z[0:N].copy()
    y = z[N:2*N].copy()
    r = z[2*N:3*N].copy()
    t = z[3*N]
    w = z[3*N+1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15:
        return 1e8, np.zeros_like(z)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    r_sum_pairs = r_sum[iu]

    # All violations array
    n_pairs = len(iu[0])
    v = np.empty(n_pairs + 4*N)
    v[:n_pairs] = r_sum_pairs - d_pairs
    v[n_pairs:n_pairs+N] = r - x
    v[n_pairs+N:n_pairs+2*N] = x + r - w
    v[n_pairs+2*N:n_pairs+3*N] = r - y
    v[n_pairs+3*N:] = y + r - h

    v_max = np.max(v)
    if v_max < -1.0/beta:
        smax = 0.0
        weights = np.zeros_like(v)
    else:
        exp_v = np.exp(beta * (v - v_max))
        smax = np.log(np.sum(exp_v)) / beta + v_max
        weights = exp_v / np.sum(exp_v)

    lam = 50.0
    obj = -np.sum(r) + lam * t + 1000.0 * np.maximum(smax - t, 0.0) ** 2

    # Gradient
    grad = np.zeros_like(z)

    # grad wrt r: -1 + contribution from smax
    grad[2*N:3*N] = -1.0

    # grad wrt t
    if smax > t:
        grad[3*N] = lam + 2000.0 * (smax - t)
    else:
        grad[3*N] = lam

    # grad wrt w (via h = 2-w)
    # Boundary contributions
    # right_viol = x + r - w  -> d/dw = -1
    # top_viol = y + r - h = y + r - (2-w) = y + r - 2 + w -> d/dw = +1
    # But smax gradient flows through weights
    if v_max >= -1.0/beta:
        # Pairwise: no direct w dependency in positions, but d_pairs depends on positions only
        # Boundary:
        # right: index n_pairs+N to n_pairs+2*N, weight contributes -1 to grad_w
        grad_w_from_right = -np.sum(weights[n_pairs+N:n_pairs+2*N])
        # top: index n_pairs+3*N to end, d(y+r-h)/dw = d(y+r-2+w)/dw = +1
        grad_w_from_top = np.sum(weights[n_pairs+3*N:])
        grad[3*N+1] = grad_w_from_right + grad_w_from_top

    # grad wrt x, y
    if v_max >= -1.0/beta:
        # Pairwise
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        gp_x = np.zeros(N)
        gp_y = np.zeros(N)
        for idx in range(n_pairs):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            wt = weights[idx]
            gp_x[i_idx] += wt * ux[idx]
            gp_y[i_idx] += wt * uy[idx]
            gp_x[j_idx] -= wt * ux[idx]
            gp_y[j_idx] -= wt * uy[idx]

        # Boundary
        # left: r - x -> d/dx_i = -1
        gp_x -= weights[n_pairs:n_pairs+N]
        # right: x + r - w -> d/dx_i = +1
        gp_x += weights[n_pairs+N:n_pairs+2*N]
        # bottom: r - y -> d/dy_i = -1
        gp_y -= weights[n_pairs+2*N:n_pairs+3*N]
        # top: y + r - h -> d/dy_i = +1
        gp_y += weights[n_pairs+3*N:]

        grad[0:N] = gp_x
        grad[N:2*N] = gp_y

    # grad wrt r from violations
    if v_max >= -1.0/beta:
        # pairwise: d/d r_i = 1 for pairs involving i
        gr = np.zeros(N)
        for idx in range(n_pairs):
            gr[iu[0][idx]] += weights[idx]
            gr[iu[1][idx]] += weights[idx]
        # boundary: left r-x: +1, right x+r-w: +1, bottom r-y: +1, top y+r-h: +1
        gr += weights[n_pairs:n_pairs+N]
        gr += weights[n_pairs+N:n_pairs+2*N]
        gr += weights[n_pairs+2*N:n_pairs+3*N]
        gr += weights[n_pairs+3*N:]
        grad[2*N:3*N] += gr

    return obj, grad


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r < 1e-6):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx*dx + dy*dy < (r[i] + r[j])**2 - tol:
                return False
    return True


def _force_relax(z, max_iter=600):
    """Force-directed relaxation for initial guesses."""
    z = z.copy().astype(float)
    n = N
    dt = 0.004
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

        # Wall
        fx -= np.maximum(r - x, 0) * 80
        fx += np.maximum(x + r - w, 0) * 80
        fy -= np.maximum(r - y, 0) * 80
        fy += np.maximum(y + r - h, 0) * 80

        # Pairwise
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx*dx + dy*dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0)
        max_ov = np.max(overlap) if overlap.size else 0
        if max_ov > 1e-7:
            f = overlap * 12
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                i_idx, j_idx = iu[0][idx], iu[1][idx]
                fx[i_idx] += ux[idx] * f[idx]
                fy[i_idx] += uy[idx] * f[idx]
                fx[j_idx] -= ux[idx] * f[idx]
                fy[j_idx] -= uy[idx] * f[idx]

        if max_ov < 1e-7:
            break

        force_mag = np.max(np.sqrt(fx*fx + fy*fy))
        if force_mag > 1.0:
            dt = max(1e-4, dt * 0.6)
        elif force_mag < 0.05:
            dt = min(0.008, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3*i] = np.clip(z[3*i], r[i], w - r[i])
            z[3*i+1] = np.clip(z[3*i+1], r[i], h - r[i])

    return z


def _polish_positions(z, time_budget):
    """Polish positions via force-directed method, then grow radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0008
    max_iter = int(min(4000, time_budget * 200))
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        fx -= np.maximum(r - x, 0) * 30
        fx += np.maximum(x + r - w, 0) * 30
        fy -= np.maximum(r - y, 0) * 30
        fy += np.maximum(y + r - h, 0) * 30

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx*dx + dy*dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0)
        if np.max(overlap) > 1e-9:
            f = overlap * 15
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                i_idx, j_idx = iu[0][idx], iu[1][idx]
                fx[i_idx] += ux[idx] * f[idx]
                fy[i_idx] += uy[idx] * f[idx]
                fx[j_idx] -= ux[idx] * f[idx]
                fy[j_idx] -= uy[idx] * f[idx]

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Uniform growth
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0004
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Per-circle growth
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
            d = np.sqrt(dx*dx + dy*dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-6:
            lo, hi = r[i], r_max
            for _ in range(18):
                mid = (lo + hi) / 2
                rt = r.copy()
                rt[i] = mid
                if _is_valid(x, y, rt, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3*N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _expansion_polish(z, time_budget):
    """Iterative per-circle expansion with position shifts."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    for rnd in range(40):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Max radius
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-5:
                lo, hi = r[i], r_max
                for _ in range(20):
                    mid = (lo + hi) / 2
                    rt = r.copy()
                    rt[i] = mid
                    if _is_valid(x, y, rt, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-6:
                    r[i] = lo
                    improved = True
            elif r_max <= r[i] + 1e-5:
                # Try to move
                min_dist = min(x[i]-r[i], w-x[i]-r[i], y[i]-r[i], h-y[i]-r[i])
                blocker = None
                if x[i]-r[i] <= min_dist + 1e-9:
                    blocker = 0  # left
                elif w-x[i]-r[i] <= min_dist + 1e-9:
                    blocker = 1  # right
                elif y[i]-r[i] <= min_dist + 1e-9:
                    blocker = 2  # bottom
                elif h-y[i]-r[i] <= min_dist + 1e-9:
                    blocker = 3  # top
                else:
                    for j in range(n):
                        if i == j:
                            continue
                        dx = x[i]-x[j]
                        dy = y[i]-y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        gap = d - r[i] - r[j]
                        if gap < min_dist:
                            min_dist = gap
                            blocker = j

                if blocker is not None and min_dist < 0.03:
                    mx, my = 0.0, 0.0
                    if blocker == 0:
                        mx = 1.0
                    elif blocker == 1:
                        mx = -1.0
                    elif blocker == 2:
                        my = 1.0
                    elif blocker == 3:
                        my = -1.0
                    elif isinstance(blocker, int) and blocker >= n:
                        pass
                    else:
                        j = blocker
                        dx = x[i]-x[j]
                        dy = y[i]-y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d > 1e-6:
                            mx, my = dx/d, dy/d

                    for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                        nx, ny = x[i] + mx*step, y[i] + my*step
                        if nx >= r[i] and nx <= w - r[i] and ny >= r[i] and ny <= h - r[i]:
                            valid = True
                            for j in range(n):
                                if i == j:
                                    continue
                                ddx = nx - x[j]
                                ddy = ny - y[j]
                                if ddx*ddx + ddy*ddy < (r[i]+r[j])**2 - 1e-8:
                                    valid = False
                                    break
                            if valid:
                                x[i], y[i] = nx, ny
                                improved = True
                                break
        if not improved:
            break

    # Final uniform growth
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3*N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _slsqp_constraint_refine(z, time_budget):
    """SLSQP with explicit constraints for high-precision refinement."""
    z = z.copy().astype(float)
    n = N

    def obj(zv):
        return -np.sum(zv[2::3][:n])

    bounds = []
    for _ in range(n):
        bounds.extend([(0.001, 1.99), (0.001, 1.99), (0.001, 0.5)])
    bounds.append((0.3, 1.7))

    constraints = []
    for i in range(n):
        bi = 3*i
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })

    maxiter = int(min(600, time_budget * 100))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 80})
        if _is_valid(res.x[0::3][:n], res.x[1::3][:n], res.x[2::3][:n], res.x[-1], 2.0 - res.x[-1]):
            return res.x
    except Exception:
        pass
    return None


def _lfgsb_epigraph(z, time_budget):
    """L-BFGS-B on epigraph formulation for position/radius refinement."""
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]

    # Build epigraph vector: [x, y, r, t, w]
    t_init = np.max([
        np.max(r[:, None] + r[None, :] - np.sqrt(np.maximum((x[:,None]-x[None,:])**2 + (y[:,None]-y[None,:])**2, 1e-12))),
        np.max(r - x), np.max(x + r - w), np.max(r - y), np.max(y + r - (2.0 - w))
    ])
    t_init = max(t_init, 0.0) + 0.01

    zv = np.concatenate([x, y, r, [t_init], [w]])

    bounds_e = []
    for _ in range(n):
        bounds_e.extend([(0.001, 1.99), (0.001, 1.99), (0.001, 0.5)])
    bounds_e.append((0.0, 1.0))  # t
    bounds_e.append((0.3, 1.7))  # w

    maxiter = int(min(800, time_budget * 80))
    try:
        res = minimize(_epigraph_objective, zv, args=(20.0,), method='L-BFGS-B',
                       bounds=bounds_e, jac=_epigraph_obj_grad,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        xo = res.x
        x_new = xo[0:n]
        y_new = xo[n:2*n]
        r_new = xo[2*n:3*n]
        w_new = xo[3*n+1]
        h_new = 2.0 - w_new
        if _is_valid(x_new, y_new, r_new, w_new, h_new):
            z_new = np.empty(3*N + 1)
            z_new[0::3][:n] = x_new
            z_new[1::3][:n] = y_new
            z_new[2::3][:n] = r_new
            z_new[-1] = w_new
            return z_new
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    def get_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        aspects = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.4, 0.7, 1.6, 0.65, 1.8, 0.6, 2.0, 0.55, 1.15, 0.85, 1.3, 0.78, 2.2, 0.5]
        for w0 in aspects:
            h0 = 2.0 - w0
            if h0 < 0.25 or w0 < 0.25:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.40
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(123)
        for _ in range(12):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.12)
                z[3*i] = rng.uniform(r0, w0 - r0)
                z[3*i+1] = rng.uniform(r0, h0 - r0)
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        return guesses

    guesses = get_guesses()

    for gi, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z_rel = _force_relax(z0, max_iter=500)

        # Phase A: SLSQP
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))
        try:
            res = minimize(_objective_vectorized, z_rel, method='SLSQP',
                           bounds=[(0.01, 1.99)]*(3*N) + [(0.35, 1.65)],
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 50})
            z_opt = res.x
            x_o = z_opt[0::3][:N]
            y_o = z_opt[1::3][:N]
            r_o = z_opt[2::3][:N]
            w_o = z_opt[-1]
            h_o = 2.0 - w_o
            if _is_valid(x_o, y_o, r_o, w_o, h_o):
                val = float(np.sum(r_o))
                if val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
        except Exception:
            pass

        # Phase B: L-BFGS-B epigraph on best so far
        if best_result is not None and time.time() < deadline - 20:
            lbfgs_res = _lfgsb_epigraph(best_result, deadline - 15)
            if lbfgs_res is not None:
                val = float(np.sum(lbfgs_res[2::3][:N]))
                if val > best_value:
                    best_value = val
                    best_result = lbfgs_res

    # Polish phases
    if best_result is not None:
        phases = [
            (_polish_positions, 35),
            (_expansion_polish, 25),
            (_slsqp_constraint_refine, 20),
            (_lfgsb_epigraph, 15),
            (_polish_positions, 10),
            (_expansion_polish, 8),
        ]
        for func, budget in phases:
            if time.time() > deadline - budget:
                break
            try:
                polished = func(best_result, deadline - budget)
                if polished is not None:
                    val = float(np.sum(polished[2::3][:N]))
                    if val > best_value:
                        best_value = val
                        best_result = polished
            except Exception:
                pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 0.001) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
