"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

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


def _energy(cx, cy, r, W, H):
    """Surrogate energy: wall + pairwise overlap penalties."""
    e = 0.0
    e += np.sum(np.maximum(r - cx, 0.0) ** 2)
    e += np.sum(np.maximum(cx + r - W, 0.0) ** 2)
    e += np.sum(np.maximum(r - cy, 0.0) ** 2)
    e += np.sum(np.maximum(cy + r - H, 0.0) ** 2)
    iu = np.triu_indices(N, k=1)
    dx = cx[iu[0]] - cx[iu[1]]
    dy = cy[iu[0]] - cy[iu[1]]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    rs = r[iu[0]] + r[iu[1]]
    ov = np.maximum(rs - d, 0.0)
    e += np.sum(ov * ov)
    return e


def _init_hex(w, h, n):
    """Hexagonal close-packed initial positions."""
    area = w * h
    r0 = np.sqrt(area / (n * np.pi * 2.0 / np.sqrt(3))) * 0.85
    r0 = min(r0, w / 7.0, h / 7.0)
    positions = []
    row_h = r0 * np.sqrt(3)
    col_w = 2.0 * r0
    for row in range(int(h / row_h) + 2):
        for col in range(int(w / col_w) + 2):
            if len(positions) >= n:
                break
            x = (col + 0.5) * col_w + (row % 2) * r0
            y = (row + 0.5) * row_h
            if x - r0 >= 0 and x + r0 <= w and y - r0 >= 0 and y + r0 <= h:
                positions.append((x, y, r0))
    while len(positions) < n:
        positions.append((w / 2, h / 2, r0 * 0.5))
    return positions[:n]


def _init_grid(w, h, n):
    """Grid initial positions."""
    best = None
    best_score = 0
    for rows in range(3, 9):
        for cols in range(3, 9):
            if rows * cols < n:
                continue
            cw = w / cols
            ch = h / rows
            r_est = min(cw, ch) / 2 * 0.92
            # Verify fit
            ok = True
            for i in range(n):
                row = i // cols
                col = i % cols
                if row >= rows:
                    ok = False
                    break
                x = (col + 0.5) * cw
                y = (row + 0.5) * ch
                if x - r_est < 0 or x + r_est > w or y - r_est < 0 or y + r_est > h:
                    ok = False
                    break
            if not ok:
                continue
            # Score by min pairwise distance
            min_d = float('inf')
            for i in range(n):
                ri, ci = i // cols, i % cols
                xi, yi = (ci + 0.5) * cw, (ri + 0.5) * ch
                for j in range(i + 1, n):
                    rj, cj = j // cols, j % cols
                    xj, yj = (cj + 0.5) * cw, (rj + 0.5) * ch
                    d = np.hypot(xi - xj, yi - yj)
                    min_d = min(min_d, d)
            if min_d > best_score:
                best_score = min_d
                best = (rows, cols, r_est)
    if best is None:
        cols = n
        cw = w / cols
        r_est = cw / 2 * 0.85
        return [( (i + 0.5) * cw, h / 2, r_est) for i in range(n)]
    rows, cols, r_est = best
    cw, ch = w / cols, h / rows
    pos = []
    for i in range(n):
        row, col = i // cols, i % cols
        pos.append(((col + 0.5) * cw, (row + 0.5) * ch, r_est))
    return pos


def _init_random(w, h, n, rng):
    """Random initial positions with varied radii."""
    x = rng.uniform(0.05, w - 0.05, n)
    y = rng.uniform(0.05, h - 0.05, n)
    r = rng.uniform(0.04, 0.13, n)
    return list(zip(x, y, r))


def _simulate(cx, cy, r, w, h, t_budget, dt=0.0008, n_iter=50000):
    """
    Rigid-body dynamics: mass ~ r^2, repulsive potential, impulse collision resolution.
    Returns (cx, cy, r) or None if invalid.
    """
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    r = r.astype(float).copy()
    if w < 0.2 or h < 0.2:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-6)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)

    iu = np.triu_indices(N, k=1)
    t0 = time.time()
    damping = 0.90
    wall_k = 8000.0
    pair_k = 5000.0

    for it in range(n_iter):
        if time.time() - t0 > t_budget:
            break
        prog = it / n_iter

        # Softening anneals from soft to hard
        soft = 0.6 * (1.0 - prog)

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        fx -= np.maximum(r - cx, 0.0) * wall_k
        fx += np.maximum(cx + r - w, 0.0) * wall_k
        fy -= np.maximum(r - cy, 0.0) * wall_k
        fy += np.maximum(cy + r - h, 0.0) * wall_k

        # Pair forces (vectorized)
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]

        d2p = d2[iu]
        rsp = rs[iu]
        dp = np.sqrt(np.maximum(d2p, 1e-12))
        ov = np.maximum(rsp - dp, 0.0)

        if np.max(ov) > 1e-10:
            fp = ov * pair_k / (1.0 + 8.0 * soft)
            dxp = dx[iu]
            dyp = dy[iu]
            ux = dxp / dp
            uy = dyp / dp
            np.add.at(fx, iu[0], ux * fp)
            np.add.at(fy, iu[0], uy * fp)
            np.add.at(fx, iu[1], -ux * fp)
            np.add.at(fy, iu[1], -uy * fp)

        # Integrate
        vx += dt * fx / mass
        vy += dt * fy / mass
        vx *= damping
        vy *= damping
        cx += dt * vx
        cy += dt * vy
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)

        # Impulse-based separation (multiple passes, increasing as we converge)
        n_pass = 6 + int(6 * prog)
        for _ in range(n_pass):
            dx = cx[:, None] - cx[None, :]
            dy = cy[:, None] - cy[None, :]
            d2 = dx * dx + dy * dy
            rs = r[:, None] + r[None, :]
            d2p = d2[iu]
            rsp = rs[iu]
            dp = np.sqrt(np.maximum(d2p, 1e-12))
            ov = np.maximum(rsp - dp, 0.0)
            if np.max(ov) <= 1e-9:
                break
            dxp = dx[iu]
            dyp = dy[iu]
            ux = dxp / dp
            uy = dyp / dp
            mi = mass[iu[0]]
            mj = mass[iu[1]]
            tm = mi + mj
            corr = ov * 0.5
            fi = mj / tm
            fj = mi / tm
            np.add.at(cx, iu[0], ux * corr * fi)
            np.add.at(cy, iu[0], uy * corr * fi)
            np.add.at(cx, iu[1], -ux * corr * fj)
            np.add.at(cy, iu[1], -uy * corr * fj)
            cx = np.clip(cx, r, w - r)
            cy = np.clip(cy, r, h - r)

    if not _is_valid(cx, cy, r, w, h):
        return None, None, None
    return cx, cy, r


def _expand(cx, cy, r, w, h, t_budget):
    """
    Expand radii: per-circle binary search, then joint moves, then uniform scaling.
    """
    cx = cx.astype(float).copy()
    cy = cy.astype(float).copy()
    r = r.astype(float).copy()
    t0 = time.time()
    improved = True
    rounds = 0

    while improved and time.time() - t0 < t_budget and rounds < 200:
        improved = False
        rounds += 1

        # Phase 1: per-circle max expansion via binary search
        for i in range(N):
            r_wall = min(cx[i], w - cx[i], cy[i], h - cy[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                ra = d - r[j]
                if ra < r_max:
                    r_max = ra
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    rt = r.copy()
                    rt[i] = mid
                    if _is_valid(cx, cy, rt, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        # Phase 2: try moving each circle to gain expansion room
        if not improved:
            for i in range(N):
                best_gain = 0.0
                best_xy = None
                for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                    for step in [0.05, 0.03, 0.015, 0.007, 0.003]:
                        nx = cx[i] + np.cos(ang) * step
                        ny = cy[i] + np.sin(ang) * step
                        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                            continue
                        ok = True
                        for j in range(N):
                            if i == j:
                                continue
                            if (nx - cx[j]) ** 2 + (ny - cy[j]) ** 2 < (r[i] + r[j]) ** 2 - 1e-9:
                                ok = False
                                break
                        if not ok:
                            continue
                        rw = min(nx, w - nx, ny, h - ny)
                        rm = rw
                        for j in range(N):
                            if i == j:
                                continue
                            d = np.hypot(nx - cx[j], ny - cy[j])
                            ra = d - r[j]
                            if ra < rm:
                                rm = ra
                        gain = max(0.0, rm - r[i])
                        if gain > best_gain:
                            best_gain = gain
                            best_xy = (nx, ny)
                if best_xy is not None and best_gain > 1e-6:
                    cx[i] = best_xy[0]
                    cy[i] = best_xy[1]
                    improved = True

        # Phase 3: pair moves
        if not improved:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    d = np.hypot(dx, dy)
                    if d < 1e-6:
                        continue
                    ux, uy = dx / d, dy / d
                    for step in [0.012, 0.006, 0.003]:
                        nxi = cx[i] + ux * step
                        nyi = cy[i] + uy * step
                        nxj = cx[j] - ux * step
                        nyj = cy[j] - uy * step
                        if (nxi < r[i] or nxi > w - r[i] or nyi < r[i] or nyi > h - r[i] or
                                nxj < r[j] or nxj > w - r[j] or nyj < r[j] or nyj > h - r[j]):
                            continue
                        ok = True
                        for k in range(N):
                            if k == i or k == j:
                                continue
                            if (nxi - cx[k]) ** 2 + (nyi - cy[k]) ** 2 < (r[i] + r[k]) ** 2 - 1e-9:
                                ok = False
                                break
                            if (nxj - cx[k]) ** 2 + (nyj - cy[k]) ** 2 < (r[j] + r[k]) ** 2 - 1e-9:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Test if this helps
                        ct = cx.copy()
                        yt = cy.copy()
                        ct[i], yt[i] = nxi, nyi
                        ct[j], yt[j] = nxj, nyj
                        rt = r.copy()
                        for _ in range(10):
                            rt2 = rt * 1.002
                            if _is_valid(ct, yt, rt2, w, h):
                                rt = rt2
                            else:
                                break
                        if np.sum(rt) > np.sum(r) + 1e-6:
                            cx[i], cy[i] = nxi, nyi
                            cx[j], cy[j] = nxj, nyj
                            r = rt
                            improved = True
                            break

    # Uniform scaling
    for _ in range(8000):
        if time.time() - t0 > t_budget:
            break
        rt = r * 1.0002
        if _is_valid(cx, cy, rt, w, h):
            r = rt
        else:
            break

    # Final per-circle expansion
    for i in range(N):
        if time.time() - t0 > t_budget:
            break
        r_wall = min(cx[i], w - cx[i], cy[i], h - cy[i])
        r_max = r_wall
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            ra = d - r[j]
            if ra < r_max:
                r_max = ra
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-7:
            lo, hi = r[i], r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                rt = r.copy()
                rt[i] = mid
                if _is_valid(cx, cy, rt, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-8:
                r[i] = lo

    if _is_valid(cx, cy, r, w, h):
        return cx, cy, r
    return None, None, None


def _opt_aspect(cx, cy, r, w0, h0, t_budget):
    """Optimize aspect ratio by trying different widths."""
    best_w = w0
    best_val = np.sum(r)
    t0 = time.time()

    for w in np.linspace(0.3, 1.7, 50):
        if time.time() - t0 > t_budget:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        sx, sy = w / w0, h / h0
        xn, yn = cx * sx, cy * sy
        rn = r * min(sx, sy)
        if _is_valid(xn, yn, rn, w, h):
            v = np.sum(rn)
            if v > best_val:
                best_val = v
                best_w = w

    for w in np.linspace(max(0.25, best_w - 0.15), min(1.75, best_w + 0.15), 40):
        if time.time() - t0 > t_budget:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        sx, sy = w / w0, h / h0
        xn, yn = cx * sx, cy * sy
        rn = r * min(sx, sy)
        if _is_valid(xn, yn, rn, w, h):
            v = np.sum(rn)
            if v > best_val:
                best_val = v
                best_w = w

    for w in np.linspace(max(0.25, best_w - 0.04), min(1.75, best_w + 0.04), 25):
        if time.time() - t0 > t_budget:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        sx, sy = w / w0, h / h0
        xn, yn = cx * sx, cy * sy
        rn = r * min(sx, sy)
        if _is_valid(xn, yn, rn, w, h):
            v = np.sum(rn)
            if v > best_val:
                best_val = v
                best_w = w

    return best_w


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    best_val = 0.0
    best_res = None

    def _valid(x, y, r, w, h, tol=1e-9):
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
                if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < (r[i] + r[j]) ** 2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    # Generate diverse initial guesses
    rng = np.random.default_rng(123)
    guesses = []
    aspects = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.25, 0.7, 0.9, 1.1, 0.8, 1.15, 0.72, 1.28, 0.65, 1.35]
    for w0 in aspects:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        # Hex
        pos = _init_hex(w0, h0, N)
        if len(pos) >= N:
            guesses.append((np.array([p[0] for p in pos]), np.array([p[1] for p in pos]),
                            np.array([p[2] for p in pos]), w0, h0))
        # Grid
        pos = _init_grid(w0, h0, N)
        if len(pos) >= N:
            guesses.append((np.array([p[0] for p in pos]), np.array([p[1] for p in pos]),
                            np.array([p[2] for p in pos]), w0, h0))

    # Random starts
    for _ in range(40):
        w0 = rng.uniform(0.45, 1.55)
        h0 = 2.0 - w0
        pos = _init_random(w0, h0, N, rng)
        guesses.append((np.array([p[0] for p in pos]), np.array([p[1] for p in pos]),
                        np.array([p[2] for p in pos]), w0, h0))

    # Sort by energy
    guesses.sort(key=lambda g: _energy(g[0], g[1], g[2], g[3], g[4]))

    for x0, y0, r0, w0, h0 in guesses:
        if time.time() > deadline - 20:
            break

        # Simulate
        rem = deadline - (time.time() - t0)
        sim_t = min(3.5, rem * 0.25)
        xs, ys, rs = _simulate(x0, y0, r0, w0, h0, sim_t)
        if xs is None:
            continue

        # Aspect optimization
        rem = deadline - (time.time() - t0)
        asp_t = min(1.5, rem * 0.1)
        wo = _opt_aspect(xs, ys, rs, w0, h0, asp_t)
        if abs(wo - w0) > 1e-6:
            ho = 2.0 - wo
            sx, sy = wo / w0, ho / h0
            xs = xs * sx
            ys = ys * sy
            rs = rs * min(sx, sy)
            w0, h0 = wo, ho
            # Short re-sim
            st = min(0.8, rem * 0.04)
            xr, yr, rr = _simulate(xs, ys, rs, w0, h0, st)
            if xr is not None:
                xs, ys, rs = xr, yr, rr

        # Expand
        rem = deadline - (time.time() - t0)
        exp_t = min(12.0, rem * 0.5)
        xe, ye, re = _expand(xs, ys, rs, w0, h0, exp_t)
        if xe is not None:
            ok, v = _valid(xe, ye, re, w0, h0)
            if ok and v > best_val:
                best_val = v
                best_res = (xe.copy(), ye.copy(), re.copy(), w0, h0)
        else:
            ok, v = _valid(xs, ys, rs, w0, h0)
            if ok and v > best_val:
                best_val = v
                best_res = (xs.copy(), ys.copy(), rs.copy(), w0, h0)

    # Final polish
    if best_res is not None and time.time() < deadline - 3:
        xb, yb, rb, wb, hb = best_res
        ft = max(2.0, deadline - (time.time() - t0) - 2)
        xf, yf, rf = _expand(xb, yb, rb, wb, hb, ft)
        if xf is not None:
            ok, v = _valid(xf, yf, rf, wb, hb)
            if ok and v > best_val:
                best_val = v
                best_res = (xf.copy(), yf.copy(), rf.copy(), wb, hb)

        # Aspect re-opt
        if time.time() < deadline - 2:
            xb, yb, rb, wb, hb = best_res
            wo2 = _opt_aspect(xb, yb, rb, wb, hb, 1.0)
            if abs(wo2 - wb) > 1e-6:
                ho2 = 2.0 - wo2
                sx, sy = wo2 / wb, ho2 / hb
                xb2, yb2, rb2 = xb * sx, yb * sy, rb * min(sx, sy)
                ok, v = _valid(xb2, yb2, rb2, wo2, ho2)
                if ok and v > best_val:
                    best_val = v
                    best_res = (xb2.copy(), yb2.copy(), rb2.copy(), wo2, ho2)

        # Final expand
        if time.time() < deadline - 1:
            xb, yb, rb, wb, hb = best_res
            pt = max(1.0, deadline - (time.time() - t0) - 0.5)
            xp, yp, rp = _expand(xb, yb, rb, wb, hb, pt)
            if xp is not None:
                ok, v = _valid(xp, yp, rp, wb, hb)
                if ok and v > best_val:
                    best_val = v
                    best_res = (xp.copy(), yp.copy(), rp.copy(), wb, hb)

    if best_res is None:
        # Fallback
        w, h = 1.0, 1.0
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / 5, (j + 0.5) / 5, 0.5 / 5 - 0.005))
        return np.array(cells[:N])

    x, y, r, w, h = best_res
    # Safety shrink
    for _ in range(20):
        rt = r * (1 - 1e-8)
        if _is_valid(x, y, rt, w, h):
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
