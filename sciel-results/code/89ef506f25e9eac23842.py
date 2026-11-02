import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21


def _softplus(x, beta=100.0):
    """Numerically stable softplus (vectorized)."""
    x_beta = beta * x
    return np.where(x_beta > 0,
                    x + np.log1p(np.exp(-x_beta)) / beta,
                    np.log1p(np.exp(x_beta)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x) (vectorized)."""
    return 1.0 / (1.0 + np.exp(-beta * x))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # ---------- vectorized validity / value ----------
    def eval_fast(circles, w, h):
        """Return sum of radii if valid, else -inf (vectorized)."""
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(xs < rs - 1e-6) or np.any(xs > w - rs + 1e-6) or \
           np.any(ys < rs - 1e-6) or np.any(ys > h - rs + 1e-6):
            return -np.inf
        # pairwise
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        need = rs[:, None] + rs[None, :]
        if np.any(d < need - 1e-6):
            return -np.inf
        return float(np.sum(rs))

    def wh_from_aspect(a):
        return 2.0 * a / (1.0 + a), 2.0 / (1.0 + a)

    # ---------- initial layouts ----------
    def hex_lattice_solution():
        """Generate hexagonal lattice packing for several aspects; return best."""
        best = None
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
        ]
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            for aspect in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 0.7, 1.4, 0.6]:
                w, h = wh_from_aspect(aspect)
                n_rows = len(row_counts)
                max_cols = max(row_counts)
                est_r_h = w / (max_cols + 0.5)
                est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1)
                r_est = min(est_r_h, est_r_v)

                circles = []
                for row_idx, n_cols in enumerate(row_counts):
                    y = (row_idx + 0.5) * r_est * np.sqrt(3)
                    row_width = n_cols * 2 * r_est
                    center_x = w / 2
                    for col_idx in range(n_cols):
                        x = center_x - row_width / 2 + (col_idx + 0.5) * 2 * r_est
                        circles.append([x, y, r_est])
                circles = np.array(circles)
                val = eval_fast(circles, w, h)
                if val > (best[0] if best else -np.inf):
                    best = (val, circles.copy(), w, h)

        if best is None:
            w, h = 1.0, 1.0
            circles = []
            for i in range(N):
                c = i % 5
                r = i // 5
                circles.append([0.1 + c * 0.2, 0.1 + r * 0.2, 0.09])
            circles = np.array(circles)
            best = (eval_fast(circles, w, h), circles, w, h)
        return best[1], best[2], best[3], best[0]

    def grid_solution(aspect, seed=0):
        rng = np.random.RandomState(seed)
        w, h = wh_from_aspect(aspect)
        cols = 5
        rows = 5
        cell_w, cell_h = w / cols, h / rows
        circles = []
        for i in range(N):
            c = i % cols
            r = i // cols
            circles.append([(c + 0.5) * cell_w, (r + 0.5) * cell_h,
                            0.5 * min(cell_w, cell_h) - 1e-4])
        circles = np.array(circles)
        val = eval_fast(circles, w, h)
        return circles, w, h, val

    # ---------- smooth optimizer (aspect is a variable) ----------
    def optimize(circles0, aspect0, max_iter=300, beta=100.0):
        """Minimize penalty - sum(r) with aspect ratio as an extra variable."""
        x0 = np.concatenate([circles0.flatten(), [aspect0]])

        def objective_and_grad(x):
            c = x[:3 * N].reshape(-1, 3)
            a = x[3 * N]
            a = min(max(a, 0.2), 5.0)
            w, h = wh_from_aspect(a)
            xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]

            val = -np.sum(rs)
            penalty = 0.0
            grad = np.zeros_like(x)

            # boundary constraints (softplus)
            v1 = rs - xs            # x >= r
            v2 = xs + rs - w        # x <= w - r
            v3 = rs - ys            # y >= r
            v4 = ys + rs - h        # y <= h - r
            v5 = -rs                # r >= 0
            for v in (v1, v2, v3, v4, v5):
                penalty += float(np.sum(_softplus(v, beta)))
            g1 = _softplus_grad(v1, beta)
            g2 = _softplus_grad(v2, beta)
            g3 = _softplus_grad(v3, beta)
            g4 = _softplus_grad(v4, beta)
            g5 = _softplus_grad(v5, beta)
            grad[:N, 0] += -g1 + g2
            grad[:N, 1] += -g3 + g4
            grad[:N, 2] += g1 + g2 + g3 + g4 - g5

            # pairwise non-overlap (softplus)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy + 1e-12)
                    v = rs[i] + rs[j] - d
                    penalty += float(_softplus(v, beta))
                    g = float(_softplus_grad(v, beta))
                    grad[i, 0] += -g * dx / d
                    grad[i, 1] += -g * dy / d
                    grad[i, 2] += g
                    grad[j, 0] += g * dx / d
                    grad[j, 1] += g * dy / d
                    grad[j, 2] += g

            # aspect gradient: d penalty / da via d(w,h)/da
            # w = 2a/(1+a) -> dw/da = 2/(1+a)^2 ; h = 2/(1+a) -> dh/da = -2/(1+a)^2
            dw = 2.0 / (1.0 + a) ** 2
            dh = -2.0 / (1.0 + a) ** 2
            # only boundary terms depend on w,h
            ga = 0.0
            ga += float(np.sum(g2 * dw))     # v2 = xs+rs-w
            ga += float(np.sum(g4 * dh))     # v4 = ys+rs-h
            grad[3 * N] = ga

            # objective: val + penalty ; d(val)/dr = -1
            grad[:N, 2] += -1.0

            return val + penalty, grad

        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        circles = result.x[:3 * N].reshape(-1, 3)
        a = min(max(result.x[3 * N], 0.2), 5.0)
        w, h = wh_from_aspect(a)
        val = eval_fast(circles, w, h)
        return circles, a, val

    # ---------- fast simulated annealing (positions, radii, aspect) ----------
    def simulated_annealing(circles0, aspect0, budget=12.0):
        random.seed(42)
        np.random.seed(42)
        circles = circles0.copy()
        aspect = float(aspect0)
        w, h = wh_from_aspect(aspect)

        def score(c, a):
            ww, hh = wh_from_aspect(a)
            return eval_fast(c, ww, hh)

        cur_score = score(circles, aspect)
        best_score = cur_score
        best_circles = circles.copy()
        best_aspect = aspect

        T = 0.03
        T_min = 1e-6
        cooling = 0.9998
        t0 = time.time()
        while time.time() - t0 < budget:
            if T < T_min:
                break
            c = circles.copy()
            a = aspect
            p = random.random()
            if p < 0.45:
                idx = random.randint(0, N - 1)
                c[idx, 0] += random.gauss(0, 0.02)
                c[idx, 1] += random.gauss(0, 0.02)
            elif p < 0.75:
                idx = random.randint(0, N - 1)
                c[idx, 2] = max(0.01, c[idx, 2] + random.gauss(0, 0.01))
            else:
                a = min(max(a * (1.0 + random.gauss(0, 0.05)), 0.2), 5.0)
                ow, oh = wh_from_aspect(aspect)
                nw, nh = wh_from_aspect(a)
                c[:, 0] *= nw / ow
                c[:, 1] *= nh / oh
            new_score = score(c, a)
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                circles, aspect = c, a
                cur_score = new_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_circles = c.copy()
                    best_aspect = a
            T *= cooling
        return best_circles, best_aspect, best_score

    # ---------- drive the search ----------
    best_value = -np.inf
    best_circles = None
    best_aspect = 1.0

    def consider(circles, aspect, val):
        nonlocal best_value, best_circles, best_aspect
        if val > best_value:
            best_value = val
            best_circles = circles.copy()
            best_aspect = aspect

    # 1) hex lattice
    circles0, w0, h0, val0 = hex_lattice_solution()
    a0 = w0 / h0
    consider(circles0, a0, val0)

    # 2) smooth optimize from hex (aspect variable)
    circles, a, val = optimize(circles0, a0, max_iter=300)
    consider(circles, a, val)

    # 3) grid inits across aspects
    if time.time() < deadline - 25:
        for aspect in [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4]:
            if time.time() > deadline - 15:
                break
            cg, wg, hg, vg = grid_solution(aspect)
            consider(cg, aspect, vg)
            if time.time() < deadline - 10:
                co, ao, vo = optimize(cg, aspect, max_iter=120)
                consider(co, ao, vo)

    # 4) SA refinement from best, then polish
    if best_circles is not None and time.time() < deadline - 20:
        sa_circles, sa_aspect, sa_val = simulated_annealing(
            best_circles, best_aspect, budget=12.0)
        consider(sa_circles, sa_aspect, sa_val)
        # polish the SA result with smooth optimizer
        if time.time() < deadline - 8:
            pc, pa, pv = optimize(sa_circles, sa_aspect, max_iter=200)
            consider(pc, pa, pv)

    # 5) final high-precision polish on the overall best
    if best_circles is not None and time.time() < deadline - 5:
        pc, pa, pv = optimize(best_circles, best_aspect, max_iter=300, beta=200.0)
        consider(pc, pa, pv)

    if best_circles is None:
        circles0, w0, h0, _ = hex_lattice_solution()
        best_circles = circles0

    # safety clamp for validity
    w, h = wh_from_aspect(best_aspect)
    for i in range(N):
        r = min(best_circles[i, 2],
                best_circles[i, 0] - 1e-6, w - best_circles[i, 0] - 1e-6,
                best_circles[i, 1] - 1e-6, h - best_circles[i, 1] - 1e-6)
        best_circles[i, 2] = max(r, 0.01)

    return best_circles
