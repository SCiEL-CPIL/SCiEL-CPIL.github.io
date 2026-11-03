"""Circle packing in perimeter-4 rectangle: joint aspect-ratio Newton-Raphson
with Delaunay contact-graph topology and multi-phase softplus polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9


def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_dev = rs - r_mean
        obj += eq_weight * np.sum(r_dev * r_dev)
        grad[2::3] += -2.0 * eq_weight * r_dev

    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1; grad[2::3] += g1
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2; grad[2::3] += g2
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3; grad[2::3] += g3
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4; grad[2::3] += g4
    v5 = -rs
    obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)
        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)
        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    grad[2::3] += -1.0
    return obj, grad


def _objective_joint(x, beta=100.0, eq_weight=0.05):
    """x = [w, x0, y0, r0, x1, y1, r1, ...] with h = 2 - w."""
    w = x[0]
    h = 2.0 - w
    c = x[1:].reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    grad_w = 0.0

    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_dev = rs - r_mean
        obj += eq_weight * np.sum(r_dev * r_dev)
        grad[3::3] += -2.0 * eq_weight * r_dev

    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
    grad[1::3] -= g1; grad[3::3] += g1
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
    grad[1::3] += g2; grad[3::3] += g2
    grad_w -= np.sum(g2)
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
    grad[2::3] -= g3; grad[3::3] += g3
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
    grad[2::3] += g4; grad[3::3] += g4
    grad_w += np.sum(g4)
    v5 = -rs
    obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
    grad[3::3] -= g5

    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)
        grad[1 + 3 * i] += -np.sum(g * dx / d)
        grad[2 + 3 * i] += -np.sum(g * dy / d)
        grad[3 + 3 * i] += np.sum(g)
        idx = np.arange(i + 1, N)
        grad[1 + 3 * idx] += g * dx / d
        grad[2 + 3 * idx] += g * dy / d
        grad[3 + 3 * idx] += g

    grad[0] = grad_w
    grad[3::3] += -1.0
    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [sorted(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    n = circles.shape[0]
    xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True


def _hex_init(w, h, seed):
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [3, 4, 4, 4, 4, 2],
        [2, 4, 4, 4, 4, 3],
        [4, 4, 4, 3, 4, 2],
        [3, 3, 3, 4, 4, 4],
        [4, 4, 4, 3, 3, 3],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    rng = random.Random(seed)
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _multi_phase_polish(circles0, w, h, deadline, eq_weight=0.05):
    circles = circles0.copy()
    phases = [
        (20.0, 80),
        (50.0, 150),
        (120.0, 250),
        (300.0, 350),
        (700.0, 450),
        (1500.0, 500),
    ]
    for beta, iters in phases:
        if time.time() > deadline:
            break
        x0 = circles.flatten()
        r_max = min(w, h) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0.0, w))
            bounds.append((0.0, h))
            bounds.append((1e-6, r_max))
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, w, h, beta, eq_weight),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            )
            circles = res.x.reshape(-1, 3)
        except Exception:
            pass
    return circles


def _joint_aspect_polish(circles0, w0, h0, deadline, eq_weight=0.03):
    """Joint optimization with w as free variable."""
    x0 = np.concatenate([[w0], circles0.flatten()])
    r_max = 1.0
    bounds = [(0.45, 1.55)]
    for _ in range(N):
        bounds.append((0.0, 1.55))
        bounds.append((0.0, 1.55))
        bounds.append((1e-6, r_max))
    phases = [
        (50.0, 100),
        (200.0, 200),
        (800.0, 300),
        (3000.0, 400),
    ]
    for beta, iters in phases:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda p: _objective_joint(p, beta, eq_weight),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60},
            )
            x0 = res.x
        except Exception:
            pass
    w_new = x0[0]
    h_new = 2.0 - w_new
    circles_new = x0[1:].reshape(-1, 3)
    if _strictly_valid(circles_new, w_new, h_new):
        return circles_new, w_new, h_new
    return circles0, w0, h0


def _newton_contact_polish(circles0, w, h, deadline, max_iters=200):
    """
    Newton-Raphson on contact constraints: for each Delaunay edge that is
    (or is near) a contact, enforce d_ij = r_i + r_j. Also enforce boundary
    contacts. This is a nonlinear system solved via Gauss-Newton / Newton
    with line search.
    """
    circles = circles0.copy()
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()

    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        # Build contact set from Delaunay + boundary
        centers = np.column_stack([xs, ys])
        neighbors = _delaunay_neighbors(centers)

        # Identify active contacts: pairs where d < r_i + r_j + margin
        margin = 0.02 * max(rs)
        contacts = []
        for i in range(N):
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-15)
                if d < rs[i] + rs[j] + margin:
                    contacts.append((i, j, 'pair'))

        # Boundary contacts
        for i in range(N):
            if xs[i] - rs[i] < margin:
                contacts.append((i, -1, 'left'))
            if xs[i] + rs[i] > w - margin:
                contacts.append((i, -2, 'right'))
            if ys[i] - rs[i] < margin:
                contacts.append((i, -3, 'bottom'))
            if ys[i] + rs[i] > h - margin:
                contacts.append((i, -4, 'top'))

        if len(contacts) < 5:
            break

        # Build Jacobian and residual for the contact constraints
        # Variables: [x0..xN, y0..yN, r0..rN, w] -> 3N+1 variables
        # But we fix w here; only optimize circle positions and radii
        # Actually let's keep w fixed and solve for 3N variables
        nvar = 3 * N
        ncon = len(contacts)

        J = np.zeros((ncon, nvar))
        f = np.zeros(ncon)

        for k, (i, j, ctype) in enumerate(contacts):
            if ctype == 'pair':
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-15)
                f[k] = d - rs[i] - rs[j]
                inv_d = 1.0 / (d + 1e-15)
                J[k, 3 * i] = dx * inv_d
                J[k, 3 * i + 1] = dy * inv_d
                J[k, 3 * i + 2] = -1.0
                J[k, 3 * j] = -dx * inv_d
                J[k, 3 * j + 1] = -dy * inv_d
                J[k, 3 * j + 2] = -1.0
            elif ctype == 'left':
                f[k] = xs[i] - rs[i]
                J[k, 3 * i] = 1.0
                J[k, 3 * i + 2] = -1.0
            elif ctype == 'right':
                f[k] = xs[i] + rs[i] - w
                J[k, 3 * i] = 1.0
                J[k, 3 * i + 2] = 1.0
            elif ctype == 'bottom':
                f[k] = ys[i] - rs[i]
                J[k, 3 * i + 1] = 1.0
                J[k, 3 * i + 2] = -1.0
            elif ctype == 'top':
                f[k] = ys[i] + rs[i] - h
                J[k, 3 * i + 1] = 1.0
                J[k, 3 * i + 2] = 1.0

        # Gauss-Newton step: solve J @ delta = -f  (least squares if overdetermined)
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(J, -f, rcond=None)
        except Exception:
            break

        # Line search
        alpha = 1.0
        obj_old = -np.sum(rs)
        for _ in range(20):
            new_xs = xs + delta[0::3]
            new_ys = ys + delta[1::3]
            new_rs = rs + delta[2::3]

            # Clamp
            new_xs = np.clip(new_xs, new_rs, w - new_rs)
            new_ys = np.clip(new_ys, new_rs, h - new_rs)
            new_rs = np.maximum(new_rs, 1e-6)

            # Check validity
            valid = True
            if np.any(new_rs < 1e-6):
                valid = False
            elif np.any(new_xs - new_rs < -1e-7) or np.any(new_xs + new_rs > w + 1e-7):
                valid = False
            elif np.any(new_ys - new_rs < -1e-7) or np.any(new_ys + new_rs > h + 1e-7):
                valid = False
            else:
                for i in range(N):
                    dx = new_xs[i] - new_xs[i + 1:]
                    dy = new_ys[i] - new_ys[i + 1:]
                    d2 = dx * dx + dy * dy
                    need = new_rs[i] + new_rs[i + 1:]
                    if np.any(d2 < need * need - 1e-10):
                        valid = False
                        break

            if valid:
                obj_new = -np.sum(new_rs)
                if obj_new <= obj_old + 1e-10:
                    xs = new_xs
                    ys = new_ys
                    rs = new_rs
                    obj_old = obj_new
                    break
            alpha *= 0.5
        else:
            # No improvement; try smaller step or break
            break

        # Check convergence
        if np.max(np.abs(delta)) < 1e-10:
            break

    circles[:, 0] = xs
    circles[:, 1] = ys
    circles[:, 2] = rs

    if _strictly_valid(circles, w, h):
        return circles
    return circles0


def _greedy_radius_polish(circles, w, h, deadline):
    circles = circles.copy()
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)
    for _ in range(80):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            new_r = old_r * 1.005
            circles[i, 2] = new_r
            if _fast_valid(circles, w, h, neighbors):
                improved = True
            else:
                circles[i, 2] = old_r
        if not improved:
            break
        if _ % 10 == 0:
            centers = circles[:, :2]
            neighbors = _delaunay_neighbors(centers)
    if not _strictly_valid(circles, w, h):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    return circles


def _high_precision_polish(circles, w, h, deadline):
    if time.time() > deadline - 3:
        return circles
    x0 = circles.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, 2000.0, 0.005),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 80},
        )
        new_circles = res.x.reshape(-1, 3)
        if _strictly_valid(new_circles, w, h):
            return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140

    best_circles = None
    best_value = -np.inf
    best_w = best_h = None

    def try_update(circles, w, h):
        nonlocal best_circles, best_value, best_w, best_h
        if circles is not None and _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
                best_w, best_h = w, h
            return val
        return -np.inf

    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.9, 0.8, 1.7, 0.7, 1.8, 0.65, 1.9]
    seed_counter = 0

    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        if time.time() > deadline - 20:
            break

        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 1000),
            _random_init(w, h, seed_counter + 2000),
        ]
        seed_counter += 50

        for circles0 in inits:
            if time.time() > deadline - 15:
                break

            # Multi-phase polish
            circles = _multi_phase_polish(circles0, w, h, deadline - 5, eq_weight=0.05)
            val = try_update(circles, w, h)

            # Newton contact polish
            if time.time() < deadline - 10:
                newton_circles = _newton_contact_polish(circles, w, h, deadline - 5)
                try_update(newton_circles, w, h)

            # Greedy radius polish
            if time.time() < deadline - 5:
                greedy_circles = _greedy_radius_polish(circles, w, h, deadline - 3)
                try_update(greedy_circles, w, h)

            # Joint aspect-ratio polish
            if time.time() < deadline - 15:
                jc, jw, jh = _joint_aspect_polish(circles, w, h, deadline - 8, eq_weight=0.02)
                try_update(jc, jw, jh)

                # Newton polish on the joint result
                if time.time() < deadline - 8:
                    nc = _newton_contact_polish(jc, jw, jh, deadline - 4)
                    try_update(nc, jw, jh)

    # Final high-precision polish on the best found
    if best_circles is not None:
        if time.time() < deadline - 8:
            cg = _newton_contact_polish(best_circles, best_w, best_h, deadline - 4)
            try_update(cg, best_w, best_h)

        if time.time() < deadline - 5:
            hp = _high_precision_polish(best_circles, best_w, best_h, deadline - 2)
            try_update(hp, best_w, best_h)

        if time.time() < deadline - 2:
            final = _greedy_radius_polish(best_circles, best_w, best_h, deadline)
            try_update(final, best_w, best_h)

    if best_circles is None:
        w, h = 1.4, 0.6
        a, b = 7, 3
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / a, h / b) * 0.45
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = r
        best_circles = circles

    return best_circles
