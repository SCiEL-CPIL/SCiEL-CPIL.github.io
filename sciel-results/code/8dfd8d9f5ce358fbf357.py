"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    def evaluate(circles, w, h, tol=1e-7):
        """Return sum of radii if valid, else -inf."""
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return -np.inf
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return -np.inf
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return -np.inf
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return -np.inf
        return float(np.sum(rs))

    def delaunay_neighbors(centers):
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(i)
                nbrs[k].add(i); nbrs[k].add(j)
            return [sorted(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def fast_valid(circles, w, h, neighbors, tol=1e-8):
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(N):
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

    def softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        c = x.reshape(-1, 3)
        xs = c[:, 0]; ys = c[:, 1]; rs = c[:, 2]
        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta)); g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta)); g2 = softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys
        obj += np.sum(softplus(v3, beta)); g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta)); g4 = softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs
        obj += np.sum(softplus(v5, beta)); g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(softplus(v, beta))
            g = softplus_grad(v, beta)
            grad[3 * i] += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx] += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        # Radius equalization penalty
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        grad[2::3] += -1.0
        return obj, grad

    def optimize(circles0, w, h, max_iter=200, beta=100.0, eq_weight=0.05):
        x0 = circles0.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                          args=(w, h, beta, eq_weight),
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9,
                                   'maxls': 50})
        return result.x.reshape(-1, 3)

    def multi_phase_polish(circles0, w, h, phases=((50, 100), (100, 300), (200, 600), (400, 1200), (800, 2000))):
        circles = circles0
        for beta, iters in phases:
            if time_left() < 3.0:
                break
            circles = optimize(circles, w, h, max_iter=iters, beta=beta)
        return circles

    def high_precision_polish(circles0, w, h):
        if time_left() < 5.0:
            return circles0
        x0 = circles0.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                          args=(w, h, 500.0, 0.01),
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                                   'maxls': 100, 'maxfun': 100000})
        return result.x.reshape(-1, 3)

    def hex_lattice_init(w, h, row_counts, seed=0):
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.98
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        if not circles:
            return None
        circles = np.array(circles)
        ys = circles[:, 1]
        shift = h / 2 - (ys.min() + ys.max()) / 2
        circles[:, 1] += shift
        return circles

    def grid_init(w, h, a, b, seed=0):
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        return circles

    def random_init(w, h, seed=0):
        rng = np.random.default_rng(seed)
        r_max = min(w, h) / 2 * 0.3
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r_max, w - r_max, N)
        circles[:, 1] = rng.uniform(r_max, h - r_max, N)
        circles[:, 2] = rng.uniform(0.02, r_max, N)
        return circles

    def sa_joint_optimize(circles0, w0, max_iters=1500, seed=42):
        """Simulated annealing with joint w optimization."""
        xs = circles0[:, 0].copy(); ys = circles0[:, 1].copy(); rs = circles0[:, 2].copy()
        w = w0; h = 2.0 - w
        T = 0.08; cooling = 0.9995
        rng = np.random.default_rng(seed)

        def cost(xs, ys, rs, w, h):
            bpen = (np.sum(np.maximum(0, rs - xs) ** 2)
                    + np.sum(np.maximum(0, xs - (w - rs)) ** 2)
                    + np.sum(np.maximum(0, rs - ys) ** 2)
                    + np.sum(np.maximum(0, ys - (h - rs)) ** 2))
            overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    viol = max(0, rs[i] + rs[j] - d)
                    overlap += viol ** 2
            return bpen + overlap - np.sum(rs)

        current_cost = cost(xs, ys, rs, w, h)
        for it in range(max_iters):
            if time_left() < 10:
                break
            if rng.random() < 0.15:
                old_w = w
                old_xs, old_ys, old_rs = xs.copy(), ys.copy(), rs.copy()
                w = np.clip(w + rng.normal(0, T * 0.5), 0.4, 1.6)
                h = 2.0 - w
                for i in range(N):
                    rs[i] = np.clip(rs[i], 1e-4, min(w, h) / 2.0)
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    w, h = old_w, 2.0 - old_w
                    xs, ys, rs = old_xs, old_ys, old_rs
                    current_cost = cost(xs, ys, rs, w, h)
            else:
                idx = rng.integers(0, N)
                old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
                scale = T * 0.3
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
                rs[idx] = np.clip(rs[idx] + rng.normal(0, scale * 0.5), 1e-4, min(w, h) / 2.0)
                xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
                ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
            T *= cooling
        return np.column_stack([xs, ys, rs]), w

    def try_candidate(circles0, w, h, budget, best):
        nonlocal best_value, best_result, best_w, best_h
        if circles0 is None:
            return
        if time_left() < budget:
            return
        circles = optimize(circles0, w, h, max_iter=200, beta=100.0)
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value, best_result, best_w, best_h = val, circles.copy(), w, h
        if time_left() < budget:
            return
        circles_p = multi_phase_polish(circles, w, h)
        nbrs = delaunay_neighbors(circles_p[:, :2])
        if fast_valid(circles_p, w, h, nbrs, tol=1e-8):
            val_p = evaluate(circles_p, w, h, tol=1e-8)
            if val_p > best_value:
                best_value, best_result, best_w, best_h = val_p, circles_p.copy(), w, h

    best_value = -np.inf
    best_result = None
    best_w = 1.0
    best_h = 1.0

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
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [5, 5, 5, 5, 1],
    ]
    row_configs = [rc for rc in row_configs if sum(rc) == N]

    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]

    init_count = 0
    for aspect in aspect_ratios:
        if time_left() < 15:
            break
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        for rc in row_configs[:6]:
            if time_left() < 10:
                break
            circles0 = hex_lattice_init(w, h, rc, seed=init_count)
            init_count += 1
            try_candidate(circles0, w, h, budget=12, best=best_value)

        if time_left() < 10:
            break
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles0 = grid_init(w, h, a, b, seed=init_count)
        init_count += 1
        try_candidate(circles0, w, h, budget=12, best=best_value)

        if time_left() < 10:
            break
        circles0 = random_init(w, h, seed=init_count)
        init_count += 1
        try_candidate(circles0, w, h, budget=12, best=best_value)

    # SA-based refinement on best aspect ratio
    if best_result is not None and time_left() > 25:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=1500, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value, best_result, best_w, best_h = val, circles_sa.copy(), w_sa, h_sa
        if time_left() > 12:
            circles_p = multi_phase_polish(circles_sa, w_sa, h_sa)
            nbrs = delaunay_neighbors(circles_p[:, :2])
            if fast_valid(circles_p, w_sa, h_sa, nbrs, tol=1e-8):
                val_p = evaluate(circles_p, w_sa, h_sa, tol=1e-8)
                if val_p > best_value:
                    best_value, best_result, best_w, best_h = val_p, circles_p.copy(), w_sa, h_sa

    # Final high-precision polish
    if best_result is not None and time_left() > 8:
        circles_hp = high_precision_polish(best_result, best_w, best_h)
        nbrs_hp = delaunay_neighbors(circles_hp[:, :2])
        if fast_valid(circles_hp, best_w, best_h, nbrs_hp, tol=1e-9):
            val_hp = evaluate(circles_hp, best_w, best_h, tol=1e-9)
            if val_hp > best_value:
                best_value, best_result = val_hp, circles_hp.copy()

    # Fallback
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                if idx >= N:
                    break
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r

    return best_result
# EVOLVE-BLOCK-END
