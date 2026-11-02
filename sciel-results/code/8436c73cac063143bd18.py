import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit

    def _softplus(x, beta=10.0):
        """Numerically stable softplus."""
        if x > 1.0 / beta:
            return x
        return np.log1p(np.exp(beta * x)) / beta

    def _compute_overlap_penalty(xs, ys, rs, beta=10.0):
        """Smooth overlap penalty using softplus surrogate."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                val = min_dist - dist
                penalty += _softplus(val, beta)
        return penalty

    def _boundary_penalty(xs, ys, rs, w, h):
        """Squared boundary violation penalty."""
        pen = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                pen += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                pen += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                pen += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                pen += (ys[i] - (h - rs[i])) ** 2
        return pen

    def _smooth_objective(params, w, h, beta=10.0):
        """
        Objective: minimize penalty - sum(r).
        params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
        """
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]

        if np.any(rs <= 0):
            return 1e10 + np.sum(np.minimum(rs, 0) ** 2)

        overlap_pen = _compute_overlap_penalty(xs, ys, rs, beta)
        boundary_pen = _boundary_penalty(xs, ys, rs, w, h)

        return overlap_pen + boundary_pen - np.sum(rs)

    def _validity_margin(result, tol=1e-4):
        """Return minimum clearance (positive = valid). result is (N,3): x, y, r."""
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]
        if np.any(rs < 1e-6):
            return -1.0

        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y

        if bw + bh > 2.0 + tol:
            return -(bw + bh - 2.0)

        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                m = d - (rs[i] + rs[j])
                if m < -tol:
                    return m

        return min(bw + bh - 2.0, 0.0)

    def _scale_to_perimeter(result):
        """Scale result about its bounding-box center so perimeter <= 4."""
        xs = result[:, 0]
        ys = result[:, 1]
        rs = result[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y

        if bw + bh <= 2.0:
            return result

        scale = 2.0 / (bw + bh) * 0.999
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        new_xs = cx + (xs - cx) * scale
        new_ys = cy + (ys - cy) * scale
        new_rs = rs * scale

        return np.column_stack((new_xs, new_ys, new_rs))

    def _make_grid_packing(m, n, W, H, n_circles):
        """Create a grid packing with m rows and n columns, placing n_circles circles."""
        circles = []
        for i in range(m):
            for j in range(n):
                if len(circles) >= n_circles:
                    break
                x = (j + 0.5) * W / n
                y = (i + 0.5) * H / m
                r = min(W / (2 * n), H / (2 * m)) * 0.99
                circles.append([x, y, r])
            if len(circles) >= n_circles:
                break
        return np.array(circles)

    def _optimize_packing(circles, W, H, max_iters=2000, tol=1e-8):
        """Optimize packing using coordinate descent for radius maximization."""
        circles = circles.copy().astype(float)
        n = len(circles)

        for it in range(max_iters):
            improved = False
            for i in range(n):
                x, y, r = circles[i]

                # Try to increase radius
                max_r = min(x, W - x, y, H - y)  # Distance to walls
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
                    if dist > 0:
                        max_r = min(max_r, dist - rj)
                new_r = max(0, max_r) * 0.999
                if new_r > r + tol:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break

        return circles

    def _try_aspect_ratio(W, H, n_circles=21):
        """Try packing for given aspect ratio using grid + coordinate descent."""
        if abs(W + H - 2.0) > 1e-9:
            scale = 2.0 / (W + H)
            W *= scale
            H *= scale

        best_circles = None
        best_value = 0.0

        # Try different grid configurations
        for m in range(1, 8):
            for n in range(1, 8):
                if m * n < n_circles:
                    continue
                if m * n > n_circles * 2:  # Don't waste too many cells
                    continue

                circles = _make_grid_packing(m, n, W, H, n_circles)
                if len(circles) < n_circles:
                    continue

                # Optimize
                circles_opt = _optimize_packing(circles, W, H)
                value = np.sum(circles_opt[:, 2])

                # Check validity
                if _validity_margin(circles_opt, W, H) >= -1e-4:
                    if value > best_value:
                        best_value = value
                        best_circles = circles_opt

        return best_circles, best_value

    def _smooth_optimize(circles, W, H, maxiter=500):
        """Optimize using smooth penalty with L-BFGS-B."""
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()

        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = xs[i]
            params[3 * i + 1] = ys[i]
            params[3 * i + 2] = rs[i]

        bounds = []
        for i in range(N):
            bounds.append((1e-6, W))
            bounds.append((1e-6, H))
            bounds.append((1e-6, min(W, H) / 2.0))

        def obj(p):
            return _smooth_objective(p, W, H, beta=10.0)

        try:
            res = minimize(obj, params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e6:  # Reasonable result
                new_xs = res.x[0::3]
                new_ys = res.x[1::3]
                new_rs = res.x[2::3]
                return np.column_stack((new_xs, new_ys, new_rs))
        except Exception:
            pass

        return circles

    def _slsqp_refine(circles, W, H, maxiter=300):
        """Refine using SLSQP with smooth objective and gradient."""
        xs = circles[:, 0].copy()
        ys = circles[:, 1].copy()
        rs = circles[:, 2].copy()

        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = xs[i]
            params[3 * i + 1] = ys[i]
            params[3 * i + 2] = rs[i]

        bounds = []
        for i in range(N):
            bounds.append((1e-6, W))
            bounds.append((1e-6, H))
            bounds.append((1e-6, min(W, H) / 2.0))

        def obj(p):
            return _smooth_objective(p, W, H, beta=20.0)

        def grad(p):
            eps = 1e-7
            g = np.zeros_like(p)
            for k in range(len(p)):
                p_plus = p.copy()
                p_plus[k] += eps
                p_minus = p.copy()
                p_minus[k] -= eps
                g[k] = (obj(p_plus) - obj(p_minus)) / (2 * eps)
            return g

        try:
            res = minimize(obj, params, method='SLSQP', jac=grad, bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.fun < 1e6:
                new_xs = res.x[0::3]
                new_ys = res.x[1::3]
                new_rs = res.x[2::3]
                return np.column_stack((new_xs, new_ys, new_rs))
        except Exception:
            pass

        return circles

    def _polish(circles, W, H):
        """Final high-precision polish pass."""
        circles = circles.copy()
        for _ in range(50):
            improved = False
            for i in range(N):
                x, y, r = circles[i]
                max_r = min(x, W - x, y, H - y)
                for j in range(N):
                    if i == j:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
                    if dist > 0:
                        max_r = min(max_r, dist - rj)
                new_r = max(0, max_r) * 0.9999
                if new_r > r + 1e-10:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break
        return circles

    best_circles = None
    best_value = 0.0
    best_W, best_H = 1.0, 1.0

    # Try various aspect ratios
    aspect_ratios = [
        (1.0, 1.0),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.8, 0.2),
        (0.2, 1.8),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
    ]

    # Phase 1: Grid + coordinate descent for multiple aspect ratios
    for W, H in aspect_ratios:
        if time.time() > deadline - 30:
            break

        circles, value = _try_aspect_ratio(W, H, N)

        if circles is not None and value > best_value:
            best_value = value
            best_circles = circles
            best_W, best_H = W, H

    # Phase 2: Smooth optimization on best result and a few promising aspect ratios
    if best_circles is not None:
        if time.time() < deadline - 20:
            refined = _smooth_optimize(best_circles, best_W, best_H, maxiter=300)
            refined_val = np.sum(refined[:, 2])
            if _validity_margin(refined, best_W, best_H) >= -1e-4 and refined_val > best_value:
                best_value = refined_val
                best_circles = refined

    # Phase 3: SLSQP refinement
    if best_circles is not None and time.time() < deadline - 10:
        refined = _slsqp_refine(best_circles, best_W, best_H, maxiter=200)
        refined_val = np.sum(refined[:, 2])
        if _validity_margin(refined, best_W, best_H) >= -1e-4 and refined_val > best_value:
            best_value = refined_val
            best_circles = refined

    # Phase 4: Final polish
    if best_circles is not None and time.time() < deadline - 5:
        best_circles = _polish(best_circles, best_W, best_H)
        best_value = np.sum(best_circles[:, 2])

    # If we didn't find anything, use the seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Scale to ensure perimeter <= 4
    best_circles = _scale_to_perimeter(best_circles)

    # Final validation and cleanup
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    rs = best_circles[:, 2]

    # Ensure circles are inside
    for i in range(N):
        if xs[i] < 0 or xs[i] > best_W or ys[i] < 0 or ys[i] > best_H or rs[i] < 0:
            # Clamp to valid region
            xs[i] = max(0.0, min(best_W, xs[i]))
            ys[i] = max(0.0, min(best_H, ys[i]))
            rs[i] = max(0.0, min(xs[i], best_W - xs[i], ys[i], best_H - ys[i]))

    # Check and fix overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if dist < rs[i] + rs[j] - 1e-9:
                # Scale down both radii
                factor = dist / (rs[i] + rs[j]) * 0.99
                rs[i] *= factor
                rs[j] *= factor
                rs[i] = max(0, min(rs[i], min(xs[i], best_W - xs[i], ys[i], best_H - ys[i])))
                rs[j] = max(0, min(rs[j], min(xs[j], best_W - xs[j], ys[j], best_H - ys[j])))

    result = np.column_stack((xs, ys, rs))
    return result
