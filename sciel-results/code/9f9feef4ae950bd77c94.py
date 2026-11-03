import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0

def construct():
    start_time = time.time()
    best_score = 0.0
    best_params = None

    pair_i, pair_j = [], []
    for i in range(N):
        for j in range(i + 1, N):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    n_pairs = len(pair_i)

    def _obj_grad(params):
        w = params[0]
        h = 2.0 - w
        if w < 0.05 or h < 0.05:
            return 1e10, np.zeros_like(params)

        x = params[1:N + 1]
        y = params[N + 1:2 * N + 1]
        r = params[2 * N + 1:3 * N + 1]

        eps = 1e-4
        lam = 500.0

        dx = x[pair_i] - x[pair_j]
        dy = y[pair_i] - y[pair_j]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        v_pair = r[pair_i] + r[pair_j] - dist

        bx = r - x
        bx2 = w - r - x
        by = r - y
        by2 = h - r - y

        def sm(u):
            return 0.5 * (u + np.sqrt(u * u + eps * eps))

        def dsm(u):
            return 0.5 * (1.0 + u / np.sqrt(u * u + eps * eps))

        sm_p = sm(v_pair)
        sm_bx1 = sm(bx)
        sm_bx2 = sm(bx2)
        sm_by1 = sm(by)
        sm_by2 = sm(by2)

        val = -np.sum(r) + lam * (np.sum(sm_p ** 2) + np.sum(sm_bx1 ** 2) + np.sum(sm_bx2 ** 2) + np.sum(sm_by1 ** 2) + np.sum(sm_by2 ** 2))

        grad = np.zeros(3 * N + 1)

        dsm_bx2 = dsm(bx2)
        dsm_by2 = dsm(by2)
        grad[0] = lam * (np.sum(2 * sm_bx2 * dsm_bx2) - np.sum(2 * sm_by2 * dsm_by2))

        dsm_bx1 = dsm(bx)
        dsm_p = dsm(v_pair)

        for i in range(N):
            grad[1 + i] = lam * (-2 * sm_bx1[i] * dsm_bx1[i] - 2 * sm_bx2[i] * dsm_bx2[i])
            for k in range(n_pairs):
                if pair_i[k] == i:
                    grad[1 + i] += lam * 2 * sm_p[k] * dsm_p[k] * (-dx[k] / dist[k])
                elif pair_j[k] == i:
                    grad[1 + i] += lam * 2 * sm_p[k] * dsm_p[k] * (dx[k] / dist[k])

        dsm_by1 = dsm(by)
        for i in range(N):
            grad[N + 1 + i] = lam * (-2 * sm_by1[i] * dsm_by1[i] - 2 * sm_by2[i] * dsm_by2[i])
            for k in range(n_pairs):
                if pair_i[k] == i:
                    grad[N + 1 + i] += lam * 2 * sm_p[k] * dsm_p[k] * (-dy[k] / dist[k])
                elif pair_j[k] == i:
                    grad[N + 1 + i] += lam * 2 * sm_p[k] * dsm_p[k] * (dy[k] / dist[k])

        for i in range(N):
            grad[2 * N + 1 + i] = -1.0 + lam * (2 * sm_bx1[i] * dsm_bx1[i] - 2 * sm_bx2[i] * dsm_bx2[i] + 2 * sm_by1[i] * dsm_by1[i] - 2 * sm_by2[i] * dsm_by2[i])
            for k in range(n_pairs):
                if pair_i[k] == i or pair_j[k] == i:
                    grad[2 * N + 1 + i] += lam * 2 * sm_p[k] * dsm_p[k]

        return val, grad

    def _validate(x, y, r, w, h):
        tol = 1e-5
        if w < 0.01 or h < 0.01:
            return False
        for i in range(N):
            if r[i] < -tol:
                return False
            if r[i] > x[i] + tol or r[i] > w - x[i] + tol:
                return False
            if r[i] > y[i] + tol or r[i] > h - y[i] + tol:
                return False
        for k in range(n_pairs):
            i, j = pair_i[k], pair_j[k]
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if r[i] + r[j] > d + tol:
                return False
        return True

    bounds = [(0.25, 1.75)]
    for _ in range(N):
        bounds.append((0.0, 2.0))
    for _ in range(N):
        bounds.append((0.0, 2.0))
    for _ in range(N):
        bounds.append((0.0, 0.95))

    aspect_ratios = [0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 12:
            break

        w0 = 2.0 * ar / (1.0 + ar)
        h0 = 2.0 / (1.0 + ar)
        if w0 < 0.25 or w0 > 1.75:
            continue

        side = 5
        for attempt in range(4):
            if time.time() - start_time > DEADLINE - 8:
                break

            cells = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    x0 = cx * w0
                    y0 = cy * h0
                    r0 = 0.42 * min(w0, h0) / side
                    cells.append([x0, y0, r0])

            np.random.shuffle(cells)
            sel = cells[:N]
            x0 = np.array([c[0] for c in sel])
            y0 = np.array([c[1] for c in sel])
            r0 = np.array([c[2] for c in sel])

            x0 = np.clip(x0 + np.random.normal(0, 0.015, N), 0.01, w0 - 0.01)
            y0 = np.clip(y0 + np.random.normal(0, 0.015, N), 0.01, h0 - 0.01)
            r0 = np.clip(r0 + np.random.normal(0, 0.003, N), 0.005, 0.4)

            params0 = np.concatenate(([w0], x0, y0, r0))

            try:
                res = minimize(
                    _obj_grad, params0, method='L-BFGS-B',
                    jac=True, bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                p = res.x
                w, xo, yo, ro = p[0], p[1:N + 1], p[N + 1:2 * N + 1], p[2 * N + 1:3 * N + 1]
                h = 2.0 - w

                if _validate(xo, yo, ro, w, h):
                    s = np.sum(ro)
                    if s > best_score:
                        best_score = s
                        best_params = (xo.copy(), yo.copy(), ro.copy())

            except Exception:
                pass

    if best_params is None:
        side = 5
        w, h = 1.0, 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                cx, cy = (i + 0.5) / side, (j + 0.5) / side
                cells.append([cx * w, cy * h, 0.42 / side])
        np.random.shuffle(cells)
        x = np.array([c[0] for c in cells[:N]])
        y = np.array([c[1] for c in cells[:N]])
        r = np.array([c[2] for c in cells[:N]])
        return np.column_stack((x, y, r))

    x, y, r = best_params
    return np.column_stack((x, y, r))
