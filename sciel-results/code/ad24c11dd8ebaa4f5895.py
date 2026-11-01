import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 21


def construct():
    t0 = time.time()
    deadline = t0 + 143.0

    def is_valid(x, y, r, w, tol=1e-7):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def delaunay_pairs(x, y):
        pts = np.column_stack([x, y])
        try:
            tri = Delaunay(pts)
        except Exception:
            return []
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    if i == j:
                        continue
                    edges.add((min(i, j), max(i, j)))
        return list(edges)

    def kkt_step(x, y, r, w, edges, lam=1e-4, max_newton=60):
        h = 2.0 - w
        n = 3 * N + 1
        A = np.zeros((n, n))
        b = np.zeros(n)
        # objective: maximize sum r  -> minimize -sum r
        for i in range(N):
            A[2 * N + i, 2 * N + i] = -1.0
        # linearized pair constraints: 2*(xi-xj)*dx + 2*(yi-yj)*dy - 2*(ri+rj) >= 0
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            ux, uy = dx / d, dy / d
            A[i, i] += 2 * ux
            A[i, N + i] += 2 * uy
            A[i, 2 * N + i] += -2.0
            A[j, j] += -2 * ux
            A[j, N + j] += -2 * uy
            A[j, 2 * N + j] += -2.0
            b[i] += 2.0 * (r[i] + r[j])
        # wall constraints (linearized, exact since linear)
        for i in range(N):
            A[N + i, i] += 1.0
            A[N + i, N + i] += 1.0
            A[N + i, 2 * N + i] += -1.0
            b[N + i] += r[i]
            A[2 * N + i, i] += -1.0
            A[2 * N + i, N + i] += -1.0
            A[2 * N + i, 2 * N + i] += 1.0
            b[2 * N + i] += -(w - r[i])
            A[3 * N + i, N + i] += 1.0
            A[3 * N + i, 2 * N + i] += -1.0
            b[3 * N + i] += r[i]
            A[3 * N + 1, N + i] += -1.0
            A[3 * N + 1, 3 * N + 1] += 1.0
            b[3 * N + 1] += h - r[i]
        # add small regularization for numerical stability
        A = A + lam * np.eye(n)
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                return x.copy(), y.copy(), r.copy(), w
        # Newton line search with backtracking
        alpha = 1.0
        best = None
        for _ in range(max_newton):
            xn = x + alpha * d[:N]
            yn = y + alpha * d[N:2 * N]
            rn = r + alpha * d[2 * N:3 * N]
            wn = w + alpha * d[3 * N]
            if wn < 0.3 or wn > 1.7:
                alpha *= 0.5
                continue
            hn = 2.0 - wn
            ok = True
            for i in range(N):
                if rn[i] <= 1e-6:
                    ok = False
                    break
                if xn[i] < rn[i] or xn[i] > wn - rn[i]:
                    ok = False
                    break
                if yn[i] < rn[i] or yn[i] > hn - rn[i]:
                    ok = False
                    break
            if ok:
                for (i, j) in edges:
                    dx = xn[i] - xn[j]
                    dy = yn[i] - yn[j]
                    if dx * dx + dy * dy < (rn[i] + rn[j]) ** 2 - 1e-8:
                        ok = False
                        break
            if ok:
                best = (xn.copy(), yn.copy(), rn.copy(), wn)
                break
            alpha *= 0.5
            if alpha < 1e-6:
                break
        if best is not None:
            return best
        return x.copy(), y.copy(), r.copy(), w

    def qpo_refine(x, y, r, w, max_rounds=60):
        x, y, r, w = x.copy(), y.copy(), r.copy(), w
        prev = -np.inf
        for _ in range(max_rounds):
            if time.time() > deadline - 6:
                break
            edges = delaunay_pairs(x, y)
            if len(edges) == 0:
                break
            xn, yn, rn, wn = kkt_step(x, y, r, w, edges)
            if not is_valid(xn, yn, rn, wn):
                break
            val = sum_radii(rn)
            if val <= prev + 1e-12:
                x, y, r, w = xn, yn, rn, wn
                break
            x, y, r, w = xn, yn, rn, wn
            prev = val
        return x, y, r, w

    def force_relax(x, y, r, w, max_iter=600):
        h = 2.0 - w
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        dt = 0.004
        for _ in range(max_iter):
            if time.time() > deadline - 6:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            max_ov = 0.0
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 150.0
                    max_ov = max(max_ov, r[i] - x[i])
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 150.0
                    max_ov = max(max_ov, x[i] - (w - r[i]))
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 150.0
                    max_ov = max(max_ov, r[i] - y[i])
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 150.0
                    max_ov = max(max_ov, y[i] - (h - r[i]))
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        fx[i] += (dx / d) * ov * 25.0
                        fy[i] += (dy / d) * ov * 25.0
                        fx[j] -= (dx / d) * ov * 25.0
                        fy[j] -= (dy / d) * ov * 25.0
                        max_ov = max(max_ov, ov)
            if max_ov < 1e-10:
                break
            fm = np.max(np.sqrt(fx * fx + fy * fy))
            if fm > 1.0:
                dt = max(1e-5, dt * 0.5)
            elif fm < 0.05:
                dt = min(0.01, dt * 1.2)
            x += dt * fx
            y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        return x, y

    def grid_init(w0):
        h0 = 2.0 - w0
        best = None
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < N or rows * cols > N + 8:
                    continue
                cw = w0 / cols
                ch = h0 / rows
                rc = min(cw, ch) / 2.0 * 0.97
                if rc < 0.02:
                    continue
                x = np.zeros(N)
                y = np.zeros(N)
                r = np.full(N, rc)
                idx = 0
                for ri in range(rows):
                    for ci in range(cols):
                        if idx >= N:
                            break
                        x[idx] = (ci + 0.5) * cw
                        y[idx] = (ri + 0.5) * ch
                        idx += 1
                if idx < N:
                    for i in range(idx, N):
                        x[i] = w0 / 2.0
                        y[i] = h0 / 2.0
                        r[i] = rc * 0.5
                if best is None or sum_radii(r) > sum_radii(best[2]):
                    best = (x, y, r)
        if best is None:
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)
            for i in range(N):
                ri, ci = divmod(i, 5)
                x[i] = (ci + 0.5) * w0 / 5.0
                y[i] = (ri + 0.5) * h0 / 5.0
            best = (x, y, r)
        return best

    def hex_init(w0):
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = min(np.sqrt(area / (N * np.pi / 0.9069)), 0.14)
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, r_est)
        dx = 2.0 * r_est * 1.003
        dy = np.sqrt(3) * r_est * 1.003
        row = col = idx = 0
        while idx < N:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1
                col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx
            y[idx] = cy
            idx += 1
            col += 1
        for i in range(idx, N):
            x[i] = w0 / 2.0
            y[i] = h0 / 2.0
            r[i] = r_est * 0.5
        return x, y, r

    def slsqp_refine(x, y, r, w, time_limit=10.0):
        remaining = time_limit - (time.time() - t0)
        if remaining < 3:
            return None
        h = 2.0 - w

        def obj(z):
            xz = z[:N]
            yz = z[N:2 * N]
            rz = z[2 * N:3 * N]
            wz = z[3 * N]
            hz = 2.0 - wz
            if wz < 0.3 or hz < 0.3:
                return 1e8
            pen = 0.0
            pen += np.sum(np.maximum(rz - xz, 0.0) ** 2)
            pen += np.sum(np.maximum(xz - (wz - rz), 0.0) ** 2)
            pen += np.sum(np.maximum(rz - yz, 0.0) ** 2)
            pen += np.sum(np.maximum(yz - (hz - rz), 0.0) ** 2)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]
                    dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = rz[i] + rz[j] - d
                    if ov > 0:
                        pen += ov * ov
            return -np.sum(rz) + 300.0 * pen

        z0 = np.concatenate([x, y, r, [w]])
        bounds = []
        for i in range(N):
            bounds.append((max(1e-4, r[i] * 0.4), w - r[i] * 0.4))
        for i in range(N):
            bounds.append((max(1e-4, r[i] * 0.4), h - r[i] * 0.4))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        try:
            maxiter = int(min(400, max(60, remaining * 25)))
            res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
            xo, yo = res.x[:N], res.x[N:2 * N]
            ro, wo = res.x[2 * N:3 * N], float(res.x[3 * N])
            if np.any(ro < 0.005):
                return None
            if is_valid(xo, yo, ro, wo):
                return xo, yo, ro, wo
            xf, yf = force_relax(xo, yo, ro, wo, max_iter=150)
            if is_valid(xf, yf, ro, wo):
                return xf, yf, ro, wo
        except Exception:
            pass
        return None

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(123)

    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.45, 0.65, 1.6, 0.6, 1.15, 0.85, 1.2, 0.95]

    for w0 in aspect_ratios:
        if time.time() > deadline - 35:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        for init_fn in (grid_init, hex_init):
            if time.time() > deadline - 30:
                break
            x0, y0, r0 = init_fn(w0)
            x0 += rng.normal(0, 0.004, N)
            y0 += rng.normal(0, 0.004, N)
            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=400)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)
            if val > best_value:
                best_value = val
                best_result = (x_t.copy(), y_t.copy(), r0.copy(), w0)

            # QP contact-graph refinement (the core new strategy)
            if time.time() < deadline - 15:
                xq, yq, rq, wq = qpo_refine(x_t, y_t, r0, w0, max_rounds=50)
                if is_valid(xq, yq, rq, wq):
                    valq = sum_radii(rq)
                    if valq > best_value:
                        best_value = valq
                        best_result = (xq.copy(), yq.copy(), rq.copy(), wq)

    # SLSQP refinement from best
    if best_result is not None and time.time() < deadline - 20:
        x_cur, y_cur, r_cur, w_cur = best_result
        res = slsqp_refine(x_cur, y_cur, r_cur, w_cur, time_limit=12.0)
        if res is not None:
            xs, ys, rs, ws = res
            val = sum_radii(rs)
            if val > best_value:
                best_value = val
                best_result = (xs.copy(), ys.copy(), rs.copy(), ws)

        # QP again from SLSQP result
        if time.time() < deadline - 8:
            x_cur, y_cur, r_cur, w_cur = best_result
            xq, yq, rq, wq = qpo_refine(x_cur, y_cur, r_cur, w_cur, max_rounds=40)
            if is_valid(xq, yq, rq, wq):
                valq = sum_radii(rq)
                if valq > best_value:
                    best_value = valq
                    best_result = (xq.copy(), yq.copy(), rq.copy(), wq)

    # Random restarts with QP
    if time.time() < deadline - 25:
        for _ in range(15):
            if time.time() > deadline - 15:
                break
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)
            x0 += rng.normal(0, 0.006, N)
            y0 += rng.normal(0, 0.006, N)
            x_t, y_t = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(x_t, y_t, r0, w0):
                continue
            val = sum_radii(r0)
            if val > best_value * 0.9:
                if time.time() < deadline - 8:
                    xq, yq, rq, wq = qpo_refine(x_t, y_t, r0, w0, max_rounds=40)
                    if is_valid(xq, yq, rq, wq):
                        valq = sum_radii(rq)
                        if valq > best_value:
                            best_value = valq
                            best_result = (xq.copy(), yq.copy(), rq.copy(), wq)
                            continue
                if time.time() < deadline - 6:
                    res = slsqp_refine(x_t, y_t, r0, w0, time_limit=6.0)
                    if res is not None:
                        xs, ys, rs, ws = res
                        val = sum_radii(rs)
                        if val > best_value:
                            best_value = val
                            best_result = (xs.copy(), ys.copy(), rs.copy(), ws)

    # Final QP polish
    if best_result is not None and time.time() < deadline - 4:
        x_cur, y_cur, r_cur, w_cur = best_result
        xq, yq, rq, wq = qpo_refine(x_cur, y_cur, r_cur, w_cur, max_rounds=30)
        if is_valid(xq, yq, rq, wq):
            valq = sum_radii(rq)
            if valq > best_value:
                best_value = valq
                best_result = (xq.copy(), yq.copy(), rq.copy(), wq)

    if best_result is None:
        w0 = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.05)
        for i in range(N):
            ri, ci = divmod(i, 5)
            x[i] = (ci + 0.5) * 0.2
            y[i] = (ri + 0.5) * 0.2
        return np.column_stack((x, y, r))

    x_final, y_final, r_final, w_final = best_result

    # Safety shrink
    for _ in range(8):
        rt = r_final * (1 - 1e-10)
        if is_valid(x_final, y_final, rt, w_final):
            r_final = rt
        else:
            break

    return np.column_stack((x_final, y_final, r_final))
