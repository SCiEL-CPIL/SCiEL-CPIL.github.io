"""
Combined approach:
- Structural idea from Inspiration: Contact-graph-aware (Delaunay) optimization for speed and precision.
- Refinement machinery from Current Program: Multi-phase L-BFGS-B with aspect ratio sweep.
- New additions: Joint aspect-ratio optimization in final polish, high-beta softplus penalties,
  radius equalization pressure, and a final high-precision polish pass with tighter tolerances.
"""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None
    best_w = None
    best_h = None

    def strictly_valid(circles, w, h, tol=1e-9):
        """Strict validity check for final result."""
        n = len(circles)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        # Check bounds
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        if np.any(rs < 1e-9):
            return False

        # Check overlaps
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            d2 = dx*dx + dy*dy
            min_d2 = (rs[i] + rs[i+1:] - tol)**2
            if np.any(d2 < min_d2):
                return False
        return True

    def evaluate(circles, w, h):
        """Return sum of radii if strictly valid, else -inf."""
        if strictly_valid(circles, w, h):
            return float(np.sum(circles[:, 2]))
        return -np.inf

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result, best_w, best_h
        if time_left() < 2.0:
            return
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_w = w
            best_h = h

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        xb = beta * x
        result = np.zeros_like(x)
        mask_pos = xb > 0
        result[mask_pos] = x[mask_pos] + np.log1p(np.exp(-xb[mask_pos])) / beta
        result[~mask_pos] = np.log1p(np.exp(xb[~mask_pos])) / beta
        return result

    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        xb = beta * x
        result = np.zeros_like(x)
        mask_pos = xb > 0
        result[mask_pos] = 1.0 / (1.0 + np.exp(-xb[mask_pos]))
        exp_xb = np.exp(xb[~mask_pos])
        result[~mask_pos] = exp_xb / (1.0 + exp_xb)
        return result

    def get_delaunay_pairs(circles, w, h, margin=0.1):
        """Get relevant circle pairs using Delaunay triangulation of centers."""
        n = len(circles)
        if n < 3:
            return [(i, j) for i in range(n) for j in range(i+1, n)]

        centers = circles[:, :2]
        try:
            tri = Delaunay(centers)
            pairs_set = set()
            for simplex in tri.simplices:
                for idx in range(len(simplex)):
                    for jdx in range(idx+1, len(simplex)):
                        i, j = simplex[idx], simplex[jdx]
                        if i > j:
                            i, j = j, i
                        pairs_set.add((i, j))

            # Add close pairs that might be missed
            xs = circles[:, 0]
            ys = circles[:, 1]
            rs = circles[:, 2]
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in pairs_set:
                        continue
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if d < rs[i] + rs[j] + margin:
                        pairs_set.add((i, j))

            return list(pairs_set)
        except Exception:
            return [(i, j) for i in range(n) for j in range(i+1, n)]

    def objective_and_grad(x, w, h, beta=100.0, pairs=None, eq_weight=0.0):
        """Force-directed penalty objective with analytical gradients."""
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure: encourage radii to be close to mean
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        # Boundary penalties (vectorized)
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
        if pairs is not None:
            for (i, j) in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)

                v = rs[i] + rs[j] - d
                obj += softplus(v, beta)
                g = softplus_grad(v, beta)

                grad[3*i] += -g * dx / d
                grad[3*i+1] += -g * dy / d
                grad[3*i+2] += g

                grad[3*j] += g * dx / d
                grad[3*j+1] += g * dy / d
                grad[3*j+2] += g
        else:
            for i in range(n):
                dx = xs[i] - xs[i+1:]
                dy = ys[i] - ys[i+1:]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)

                v = rs[i] + rs[i+1:] - d
                obj += np.sum(softplus(v, beta))
                g = softplus_grad(v, beta)

                grad[3*i] += -np.sum(g * dx / d)
                grad[3*i+1] += -np.sum(g * dy / d)
                grad[3*i+2] += np.sum(g)

                idx = np.arange(i+1, n)
                grad[3*idx] += g * dx / d
                grad[3*idx+1] += g * dy / d
                grad[3*idx+2] += g

        # Gradient of -sum(r)
        grad[2::3] += -1.0

        return obj, grad

    def hex_lattice_init(w, h):
        """Generate hexagonal lattice initialization for N=21 circles."""
        best_circles = None
        best_val = -np.inf

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
        ]

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
            val = evaluate(circles, w, h)

            if val > best_val:
                best_val = val
                best_circles = circles.copy()

        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / a, h / b) * 0.45
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)

        return best_circles

    def optimize_from(circles0, w, h, max_iter=400, beta=100.0, ftol=1e-12, gtol=1e-8, use_delaunay=False, eq_weight=0.0):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()

        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))

        try:
            if use_delaunay:
                pairs = get_delaunay_pairs(circles0, w, h)
                def obj_with_pairs(p):
                    return objective_and_grad(p, w, h, beta, pairs=pairs, eq_weight=eq_weight)
                res = minimize(
                    obj_with_pairs,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
                )
                final_circles = res.x.reshape(-1, 3)
                if evaluate(final_circles, w, h) != -np.inf:
                    return res.x
                else:
                    res2 = minimize(
                        lambda p: objective_and_grad(p, w, h, beta, eq_weight=eq_weight),
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
                    )
                    return res2.x
            else:
                res = minimize(
                    lambda p: objective_and_grad(p, w, h, beta, eq_weight=eq_weight),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
                )
                return res.x
        except Exception:
            return x0

    def joint_aspect_optimize(circles0, w0, h0, max_iter=300, beta=500.0, eq_weight=0.005):
        """
        Joint optimization of circle positions/radii AND aspect ratio.
        State: [x_1, y_1, r_1, ..., x_N, y_N, r_N, log_aspect]
        Perimeter constraint: 2*(w+h) = 4 => w+h = 2, with aspect = w/h.
        w = 2*aspect/(1+aspect), h = 2/(1+aspect).
        """
        log_aspect0 = np.log(w0 / h0)
        x0 = np.concatenate([circles0.flatten(), [log_aspect0]])

        def joint_obj_and_grad(x):
            c = x[:3*N].reshape(-1, 3)
            log_aspect = x[3*N]
            aspect = np.exp(log_aspect)
            # Clamp aspect to reasonable range
            aspect = np.clip(aspect, 0.3, 3.0)
            w = 2.0 * aspect / (1.0 + aspect)
            h = 2.0 / (1.0 + aspect)

            obj_c, grad_c = objective_and_grad(x[:3*N], w, h, beta, eq_weight=eq_weight)
            obj = obj_c

            # Perimeter is fixed at 4 by construction (w+h=2 always), so no extra penalty needed.
            # But we need gradient w.r.t. log_aspect.
            # d(obj)/d(log_aspect) = d(obj)/d(w) * dw/d(log_aspect) + d(obj)/d(h) * dh/d(log_aspect)
            # dw/d(log_aspect) = dw/d(aspect) * d(aspect)/d(log_aspect)
            #   aspect = exp(log_aspect), d(aspect)/d(log_aspect) = aspect
            #   w = 2a/(1+a), dw/da = 2/(1+a)^2
            #   dw/d(log_aspect) = 2*aspect/(1+aspect)^2
            # dh/d(log_aspect): h = 2/(1+a), dh/da = -2/(1+a)^2, dh/d(log_aspect) = -2*aspect/(1+aspect)^2

            # We need d(obj)/dw and d(obj)/dh.
            # From the objective, the boundary terms give us these.
            # Alternatively, compute numerically or analytically.
            # For simplicity, use a small numerical derivative for the aspect component.
            eps = 1e-6
            la_p = log_aspect + eps
            la_m = log_aspect - eps
            a_p = np.exp(la_p); a_p = np.clip(a_p, 0.3, 3.0)
            a_m = np.exp(la_m); a_m = np.clip(a_m, 0.3, 3.0)
            w_p = 2.0 * a_p / (1.0 + a_p); h_p = 2.0 / (1.0 + a_p)
            w_m = 2.0 * a_m / (1.0 + a_m); h_m = 2.0 / (1.0 + a_m)

            obj_p, _ = objective_and_grad(x[:3*N], w_p, h_p, beta, eq_weight=eq_weight)
            obj_m, _ = objective_and_grad(x[:3*N], w_m, h_m, beta, eq_weight=eq_weight)
            grad_aspect = (obj_p - obj_m) / (2.0 * eps)

            grad = np.concatenate([grad_c, [grad_aspect]])
            return obj, grad

        bounds = []
        r_max = min(w0, h0) / 2.0
        for i in range(N):
            bounds.append((0, 2.0))
            bounds.append((0, 2.0))
            bounds.append((1e-6, 1.0))
        bounds.append((np.log(0.3), np.log(3.0)))

        try:
            res = minimize(
                joint_obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            return res.x
        except Exception:
            return x0

    # Try multiple aspect ratios with finer resolution around promising values
    aspects = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7]

    for aspect in aspects:
        if time_left() < 18:
            break

        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        circles0 = hex_lattice_init(w, h)
        check_and_update(circles0, w, h)

        # Phase 1: Coarse optimization
        if time_left() > 15:
            x1 = optimize_from(circles0, w, h, max_iter=200, beta=50.0, eq_weight=0.01)
            check_and_update(x1.reshape(-1,3), w, h)

        # Phase 2: Medium optimization with Delaunay
        if time_left() > 12:
            x2 = optimize_from(x1, w, h, max_iter=200, beta=100.0, use_delaunay=True, eq_weight=0.01)
            check_and_update(x2.reshape(-1,3), w, h)

        # Phase 3: High precision polish
        if time_left() > 9:
            c3 = x2.reshape(-1, 3)
            pairs3 = get_delaunay_pairs(c3, w, h, margin=0.05)
            def obj_phase3(p):
                return objective_and_grad(p, w, h, beta=500.0, pairs=pairs3, eq_weight=0.005)
            bounds3 = []
            r_max3 = min(w, h) / 2.0
            for i in range(N):
                bounds3.append((0, w))
                bounds3.append((0, h))
                bounds3.append((1e-6, r_max3))
            try:
                res3 = minimize(obj_phase3, x2, method='L-BFGS-B', jac=True, bounds=bounds3,
                               options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9})
                x3 = res3.x
            except Exception:
                x3 = x2
            check_and_update(x3.reshape(-1,3), w, h)

        # Phase 4: Ultra-high precision polish
        if time_left() > 6:
            c4 = x3.reshape(-1, 3)
            pairs4 = get_delaunay_pairs(c4, w, h, margin=0.03)
            def obj_phase4(p):
                return objective_and_grad(p, w, h, beta=2000.0, pairs=pairs4, eq_weight=0.001)
            bounds4 = []
            r_max4 = min(w, h) / 2.0
            for i in range(N):
                bounds4.append((0, w))
                bounds4.append((0, h))
                bounds4.append((1e-6, r_max4))
            try:
                res4 = minimize(obj_phase4, x3, method='L-BFGS-B', jac=True, bounds=bounds4,
                               options={'maxiter': 350, 'ftol': 1e-14, 'gtol': 1e-10})
                x4 = res4.x
            except Exception:
                x4 = x3
            check_and_update(x4.reshape(-1,3), w, h)

        # Phase 5: Final tight polish with full pairs
        if time_left() > 4:
            x5 = optimize_from(x4, w, h, max_iter=250, beta=5000.0, use_delaunay=False, eq_weight=0.0, ftol=1e-15, gtol=1e-12)
            check_and_update(x5.reshape(-1,3), w, h)

        # Phase 6: Micro-polish
        if time_left() > 2.5:
            c6 = x5.reshape(-1, 3)
            pairs6 = get_delaunay_pairs(c6, w, h, margin=0.01)
            def obj_phase6(p):
                return objective_and_grad(p, w, h, beta=10000.0, pairs=pairs6, eq_weight=0.0)
            bounds6 = []
            r_max6 = min(w, h) / 2.0
            for i in range(N):
                bounds6.append((0, w))
                bounds6.append((0, h))
                bounds6.append((1e-6, r_max6))
            try:
                res6 = minimize(obj_phase6, x5, method='L-BFGS-B', jac=True, bounds=bounds6,
                               options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13})
                x6 = res6.x
                check_and_update(x6.reshape(-1,3), w, h)
            except Exception:
                pass

    # Final joint aspect-ratio optimization on the best result found
    if best_result is not None and time_left() > 20:
        w_best = best_w
        h_best = best_h
        c_best = best_result

        # Joint optimization: allow aspect ratio to vary
        x_joint = joint_aspect_optimize(c_best, w_best, h_best, max_iter=300, beta=500.0, eq_weight=0.005)
        c_joint = x_joint[:3*N].reshape(-1, 3)
        log_aspect_final = x_joint[3*N]
        aspect_final = np.exp(log_aspect_final)
        aspect_final = np.clip(aspect_final, 0.3, 3.0)
        w_final = 2.0 * aspect_final / (1.0 + aspect_final)
        h_final = 2.0 / (1.0 + aspect_final)

        check_and_update(c_joint, w_final, h_final)

        # Polish the joint result with fixed aspect
        if time_left() > 10:
            pairs_j = get_delaunay_pairs(c_joint, w_final, h_final, margin=0.02)
            def obj_joint_polish(p):
                return objective_and_grad(p, w_final, h_final, beta=5000.0, pairs=pairs_j, eq_weight=0.0)
            bounds_j = []
            r_max_j = min(w_final, h_final) / 2.0
            for i in range(N):
                bounds_j.append((0, w_final))
                bounds_j.append((0, h_final))
                bounds_j.append((1e-6, r_max_j))
            try:
                res_j = minimize(obj_joint_polish, c_joint.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_j,
                                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
                x_j = res_j.x
                check_and_update(x_j.reshape(-1,3), w_final, h_final)
            except Exception:
                pass

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.099
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
