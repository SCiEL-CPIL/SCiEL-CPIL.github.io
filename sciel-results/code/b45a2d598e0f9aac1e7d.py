"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Combined approach: vectorized softplus L-BFGS with multi-phase beta annealing,
Delaunay-accelerated validity, and multi-start over aspect ratios and row configs.
"""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def objective_and_grad(x, w, h, beta=100.0):
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -float(np.sum(rs))
        grad = np.zeros_like(x)

        # Boundary penalties (vectorized)
        v1 = rs - xs
        obj += float(np.sum(softplus(v1, beta)))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += float(np.sum(softplus(v2, beta)))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += float(np.sum(softplus(v3, beta)))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += float(np.sum(softplus(v4, beta)))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += float(np.sum(softplus(v5, beta)))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties (vectorized per i)
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += float(np.sum(softplus(v, beta)))
            g = softplus_grad(v, beta)

            grad[3 * i] += -float(np.sum(g * dx / d))
            grad[3 * i + 1] += -float(np.sum(g * dy / d))
            grad[3 * i + 2] += float(np.sum(g))

            idx = np.arange(i + 1, N)
            grad[3 * idx] += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def strictly_valid(circles, w, h, tol=1e-8):
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
        return True

    def evaluate(circles, w, h, tol=1e-7):
        if strictly_valid(circles, w, h, tol):
            return float(np.sum(circles[:, 2]))
        return -np.inf

    def delaunay_neighbors(centers):
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j)
                nbrs[i].add(k)
                nbrs[j].add(i)
                nbrs[j].add(k)
                nbrs[k].add(i)
                nbrs[k].add(j)
            return [list(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def fast_valid(circles, w, h, neighbors, tol=1e-9):
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
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

    def hex_lattice_init(w, h, row_counts, seed=0):
        rng = np.random.default_rng(seed)
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            r_est = min(w, h) / 10.0

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
            circles = [[w / 2, h / 2, r_est]]

        circles = np.array(circles)
        if len(circles) < N:
            # Pad with small circles at random positions
            extra = N - len(circles)
            for _ in range(extra):
                circles = np.vstack([circles, [w / 2 + rng.normal(0, 0.1), h / 2 + rng.normal(0, 0.1), r_est * 0.3]])
        elif len(circles) > N:
            circles = circles[:N]

        # Center vertically
        ys = circles[:, 1]
        y_center = (ys.min() + ys.max()) / 2
        circles[:, 1] += h / 2 - y_center

        # Clip to valid range
        for i in range(N):
            circles[i, 2] = min(circles[i, 2], min(w, h) / 2.0)
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], w - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], h - circles[i, 2])

        return circles

    def grid_init(w, h, seed=0):
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = r
        return circles

    def random_init(w, h, seed=0):
        rng = np.random.default_rng(seed)
        r0 = min(w, h) / 2.0 * 0.25
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
        return circles

    def multi_phase_optimize(circles0, w, h, phases=None, max_iter_per_phase=200):
        """Run L-BFGS with increasing beta phases for smoother convergence."""
        if phases is None:
            phases = [(50.0, 100), (100.0, 150), (200.0, 200), (400.0, 300)]

        circles = circles0.copy()
        for beta, max_iter in phases:
            if time_left() < 10:
                break
            x0 = circles.flatten()

            def obj_grad(x):
                return objective_and_grad(x, w, h, beta)

            try:
                result = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                                  options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
                circles = result.x.reshape(-1, 3)
            except Exception:
                pass

        return circles

    def polish(circles0, w, h, max_iter=400, beta=400.0):
        """High-precision single-phase polish."""
        x0 = circles0.flatten()

        def obj_grad(x):
            return objective_and_grad(x, w, h, beta)

        try:
            result = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
            circles = result.x.reshape(-1, 3)
        except Exception:
            circles = circles0
        return circles

    def sa_joint_optimize(circles0, w0, max_iters=2000, seed=42):
        """Simulated annealing jointly optimizing w, positions, and radii."""
        xs = circles0[:, 0].copy()
        ys = circles0[:, 1].copy()
        rs = circles0[:, 2].copy()
        w = w0
        h = 2.0 - w

        T = 0.08
        cooling = 0.9995
        rng = np.random.default_rng(seed)

        def cost(xs, ys, rs, w, h):
            bpen = float(np.sum(np.maximum(0, rs - xs) ** 2) +
                         np.sum(np.maximum(0, xs - (w - rs)) ** 2) +
                         np.sum(np.maximum(0, rs - ys) ** 2) +
                         np.sum(np.maximum(0, ys - (h - rs)) ** 2))
            overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    viol = max(0, rs[i] + rs[j] - d)
                    overlap += viol ** 2
            return bpen + overlap - float(np.sum(rs))

        current_cost = cost(xs, ys, rs, w, h)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            move_type = rng.random()

            if move_type < 0.15:
                old_w = w
                old_xs, old_ys, old_rs = xs.copy(), ys.copy(), rs.copy()
                w += rng.normal(0, T * 0.5)
                w = np.clip(w, 0.4, 1.6)
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
                    w = old_w
                    h = 2.0 - w
                    xs, ys, rs = old_xs, old_ys, old_rs
                    current_cost = cost(xs, ys, rs, w, h)
            else:
                idx = int(rng.integers(0, N))
                old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
                scale = T * 0.3
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
                rs[idx] += rng.normal(0, scale * 0.5)
                rs[idx] = np.clip(rs[idx], 1e-4, min(w, h) / 2.0)
                xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
                ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r

            T *= cooling

        circles = np.column_stack([xs, ys, rs])
        return circles, w

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
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [5, 4, 5, 4, 3],
    ]

    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]

    init_count = 0
    for aspect in aspect_ratios:
        if time_left() < 15:
            break
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        # Try hex lattice inits
        for rc in row_configs[:6]:
            if time_left() < 10:
                break
            circles0 = hex_lattice_init(w, h, rc, seed=init_count)
            val = evaluate(circles0, w, h)
            if val > best_value:
                best_value = val
                best_result = circles0.copy()
                best_w, best_h = w, h

            if time_left() < 20:
                break
            circles = multi_phase_optimize(circles0, w, h)
            val = evaluate(circles, w, h)
            if val > best_value:
                best_value = val
                best_result = circles.copy()
                best_w, best_h = w, h

            if time_left() < 10:
                break
            circles_p = polish(circles, w, h, max_iter=300)
            val_p = evaluate(circles_p, w, h)
            if val_p > best_value:
                best_value = val_p
                best_result = circles_p.copy()
                best_w, best_h = w, h

            init_count += 1

        # Try grid init
        if time_left() < 10:
            break
        circles0 = grid_init(w, h, seed=init_count)
        val = evaluate(circles0, w, h)
        if val > best_value:
            best_value = val
            best_result = circles0.copy()
            best_w, best_h = w, h

        if time_left() < 15:
            break
        circles = multi_phase_optimize(circles0, w, h)
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_w, best_h = w, h

        if time_left() < 10:
            break
        circles_p = polish(circles, w, h, max_iter=300)
        val_p = evaluate(circles_p, w, h)
        if val_p > best_value:
            best_value = val_p
            best_result = circles_p.copy()
            best_w, best_h = w, h

        init_count += 1

        # Try random init
        if time_left() < 10:
            break
        circles0 = random_init(w, h, seed=init_count)
        if time_left() < 15:
            break
        circles = multi_phase_optimize(circles0, w, h)
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_w, best_h = w, h

        init_count += 1

    # SA-based refinement on best found
    if best_result is not None and time_left() < 25:
        circles_sa, w_sa = sa_joint_optimize(best_result, best_w, max_iters=1500, seed=42)
        h_sa = 2.0 - w_sa
        val = evaluate(circles_sa, w_sa, h_sa)
        if val > best_value:
            best_value = val
            best_result = circles_sa.copy()
            best_w, best_h = w_sa, h_sa

        if time_left() < 10:
            pass
        else:
            circles_p = polish(circles_sa, w_sa, h_sa, max_iter=400)
            val_p = evaluate(circles_p, w_sa, h_sa)
            if val_p > best_value:
                best_value = val_p
                best_result = circles_p.copy()
                best_w, best_h = w_sa, h_sa

    # Final polish on best result if time remains
    if best_result is not None and time_left() > 5:
        circles_final = polish(best_result, best_w, best_h, max_iter=300, beta=500.0)
        val_final = evaluate(circles_final, best_w, best_h)
        if val_final > best_value:
            best_value = val_final
            best_result = circles_final.copy()

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
