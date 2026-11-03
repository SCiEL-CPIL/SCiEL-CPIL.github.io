import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    best_value = -np.inf
    best_result = None

    def _softplus(x, beta=100.0):
        """Numerically stable softplus."""
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.02):
        """
        Softplus-penalized objective with analytical gradients.
        x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
        Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
        """
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Equalization pressure
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_var = np.mean((rs - r_mean) ** 2)
            obj += eq_weight * r_var
            grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

        # Boundary penalties
        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        # Pairwise overlap penalties
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

        # Gradient of -sum(r)
        grad[2::3] += -1.0

        return obj, grad

    def _check_valid(x, w, h, tol=1e-7):
        """Check if a flat array [x0,y0,r0,...] is valid."""
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol:
                return False
            if yi - ri < -tol or yi + ri > h + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - tol)**2:
                    return False
        return True

    def _update_best(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        if _check_valid(x, w, h):
            val = float(np.sum(x[2::3]))
            if val > best_value:
                best_value = val
                best_result = x.copy()

    def _hex_init(w, h, seed=42):
        """Hexagonal lattice init over several row configurations."""
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
            # Quick validity check
            valid = True
            for i in range(N):
                xi, yi, ri = circles[i]
                if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist_sq = (xi-xj)**2 + (yi-yj)**2
                        min_dist = ri + rj
                        if dist_sq < (min_dist - 1e-7)**2:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best = circles.copy()
        if best is None:
            # Fallback grid
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

    def _grid_init(w, h, seed=42):
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

    def _polish(circles, W, H):
        """Final high-precision polish pass."""
        circles = circles.copy()
        for _ in range(200):
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
                new_r = max(0, max_r) * 0.99999
                if new_r > r + 1e-12:
                    circles[i, 2] = new_r
                    improved = True
            if not improved:
                break
        return circles

    def _contact_polish(circles, W, H, max_time=8.0):
        """
        High-beta contact-graph-aware polish:
        1. Build contact graph from current packing
        2. Run high-beta optimization to tighten constraints
        3. Extract maximum radii given positions
        4. Repeat
        """
        circles = circles.copy()
        t_start = time.time()
        t_max = t_start + max_time

        for phase in range(6):
            if time.time() > t_max:
                break

            # Build contact graph: pairs within 1.01*(ri+rj)
            contacts = []
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2)
                    if dist < 1.01 * (circles[i,2] + circles[j,2]) + 1e-9:
                        contacts.append((i, j))

            # Also add boundary contacts
            for i in range(N):
                if circles[i,0] - circles[i,2] < 1e-6:
                    contacts.append((i, -1))  # left wall
                if circles[i,0] + circles[i,2] > W - 1e-6:
                    contacts.append((i, -2))  # right wall
                if circles[i,1] - circles[i,2] < 1e-6:
                    contacts.append((i, -3))  # bottom wall
                if circles[i,1] + circles[i,2] > H - 1e-6:
                    contacts.append((i, -4))  # top wall

            # Run high-beta optimization
            beta = 5000 if phase >= 3 else 2000
            eq_w = 0.001

            params = np.zeros(3 * N)
            for i in range(N):
                params[3*i] = circles[i, 0]
                params[3*i+1] = circles[i, 1]
                params[3*i+2] = circles[i, 2]

            bounds = []
            for i in range(N):
                bounds.append((1e-8, W))
                bounds.append((1e-8, H))
                bounds.append((1e-8, min(W, H) / 2.0))

            def obj(p):
                return _objective_and_grad(p, W, H, beta=beta, eq_weight=eq_w)[0]

            def jac(p):
                return _objective_and_grad(p, W, H, beta=beta, eq_weight=eq_w)[1]

            try:
                res = minimize(obj, params, jac=jac, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    circles = res.x.reshape(-1, 3).copy()
            except Exception:
                pass

            # Extract maximum radii
            circles = _polish(circles, W, H)

            # Verify validity
            if not _check_valid(circles.flatten(), W, H):
                break

        return circles

    def _homotopy_optimize(init_circles, W, H, max_total_time=25.0):
        """
        Homotopy (beta-annealing) optimization: start with smooth beta,
        gradually sharpen constraints, re-optimizing at each stage.
        """
        t_start = time.time()
        t_max = t_start + max_total_time

        # Beta schedule: start smooth, end sharp
        beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        eq_weights = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]

        params = np.zeros(3 * N)
        for i in range(N):
            params[3 * i] = init_circles[i, 0]
            params[3 * i + 1] = init_circles[i, 1]
            params[3 * i + 2] = init_circles[i, 2]

        best_params = params.copy()
        best_obj = np.inf

        for stage, (beta, eq_w) in enumerate(zip(beta_schedule, eq_weights)):
            if time.time() > t_max:
                break

            bounds = []
            for i in range(N):
                bounds.append((1e-8, W))
                bounds.append((1e-8, H))
                bounds.append((1e-8, min(W, H) / 2.0))

            def obj(p):
                return _objective_and_grad(p, W, H, beta=beta, eq_weight=eq_w)[0]

            def jac(p):
                return _objective_and_grad(p, W, H, beta=beta, eq_weight=eq_w)[1]

            try:
                res = minimize(obj, params, jac=jac, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x.copy()
                params = res.x.copy()
            except Exception:
                pass

            # Check validity and update
            _update_best(params, W, H)

        return best_params.reshape(-1, 3)

    def _aspect_refine(W0, H0, max_time=15.0):
        """
        Refine aspect ratio around (W0, H0) by trying small perturbations.
        For each candidate aspect, run a quick optimization and keep best.
        """
        t_start = time.time()
        t_max = t_start + max_time

        best_circles = None
        best_val = -np.inf

        # Try aspect ratio perturbations
        aspect_perturbations = [
            0.95, 0.97, 1.0, 1.03, 1.05,
            0.9, 1.1, 0.85, 1.15,
        ]

        for pert in aspect_perturbations:
            if time.time() > t_max:
                break

            # Perturb aspect ratio
            W_new = W0 * pert
            H_new = H0 / pert  # Keep area roughly constant
            # Normalize to perimeter 4
            scale = 2.0 / (W_new + H_new)
            Wn = W_new * scale
            Hn = H_new * scale

            if Wn <= 0 or Hn <= 0:
                continue

            # Quick hex init
            init = _hex_init(Wn, Hn, seed=42)

            # Quick homotopy
            remaining = max(1.0, t_max - (time.time() - t_start) - 1.0)
            result = _homotopy_optimize(init, Wn, Hn, max_total_time=remaining)
            polished = _polish(result, Wn, Hn)

            if _check_valid(polished.flatten(), Wn, Hn):
                val = float(np.sum(polished[:, 2]))
                if val > best_val:
                    best_val = val
                    best_circles = polished.copy()

        return best_circles

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

    # Aspect ratios to try
    aspect_ratios = [
        (1.0, 1.0),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.6, 0.4),
        (0.4, 1.6),
        (1.8, 0.2),
        (0.2, 1.8),
    ]

    # Phase 1: Try multiple aspect ratios with hex/grid init + homotopy
    for W, H in aspect_ratios:
        if time.time() > deadline - 30:
            break

        # Normalize to perimeter 4
        scale = 2.0 / (W + H)
        Wn = W * scale
        Hn = H * scale

        # Try hex init
        init_hex = _hex_init(Wn, Hn, seed=42)
        _update_best(init_hex.flatten(), Wn, Hn)

        # Homotopy optimization
        remaining = max(5.0, deadline - time.time() - 25.0)
        result = _homotopy_optimize(init_hex, Wn, Hn, max_total_time=remaining)
        _update_best(result.flatten(), Wn, Hn)

        # Polish
        polished = _polish(result, Wn, Hn)
        _update_best(polished.flatten(), Wn, Hn)

        # Contact polish for promising results
        if time.time() < deadline - 20:
            cp = _contact_polish(polished, Wn, Hn, max_time=5.0)
            _update_best(cp.flatten(), Wn, Hn)

    # Phase 2: Try grid init for best aspect ratio
    if best_result is not None:
        # Reconstruct W, H from best result bounding box
        xs = best_result[0::3]
        ys = best_result[1::3]
        rs = best_result[2::3]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        Wn = max_x - min_x
        Hn = max_y - min_y

        if time.time() < deadline - 25:
            # Try grid init with different seeds
            for seed in [1, 2, 3, 7, 13, 17]:
                if time.time() > deadline - 20:
                    break
                init_grid = _grid_init(Wn, Hn, seed=seed)
                _update_best(init_grid.flatten(), Wn, Hn)
                remaining = max(3.0, deadline - time.time() - 20.0)
                result = _homotopy_optimize(init_grid, Wn, Hn, max_total_time=remaining)
                _update_best(result.flatten(), Wn, Hn)
                polished = _polish(result, Wn, Hn)
                _update_best(polished.flatten(), Wn, Hn)

                # Contact polish
                if time.time() < deadline - 15:
                    cp = _contact_polish(polished, Wn, Hn, max_time=4.0)
                    _update_best(cp.flatten(), Wn, Hn)

    # Phase 3: Aspect ratio refinement around best
    if best_result is not None and time.time() < deadline - 15:
        xs = best_result[0::3]
        ys = best_result[1::3]
        rs = best_result[2::3]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        Wn = max_x - min_x
        Hn = max_y - min_y

        remaining = max(5.0, deadline - time.time() - 10.0)
        refined = _aspect_refine(Wn, Hn, max_time=remaining)
        if refined is not None:
            _update_best(refined.flatten(), Wn, Hn)

            # Final contact polish on refined result
            if time.time() < deadline - 8:
                final_cp = _contact_polish(refined, Wn, Hn, max_time=5.0)
                _update_best(final_cp.flatten(), Wn, Hn)

    # If we didn't find anything, use a default grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N]).flatten()
        best_value = float(np.sum(best_result[2::3]))

    # Convert to (N, 3) array
    result_2d = best_result.reshape(-1, 3)

    # Final contact polish before scaling
    if time.time() < deadline - 5:
        min_x = np.min(result_2d[:, 0] - result_2d[:, 2])
        max_x = np.max(result_2d[:, 0] + result_2d[:, 2])
        min_y = np.min(result_2d[:, 1] - result_2d[:, 2])
        max_y = np.max(result_2d[:, 1] + result_2d[:, 2])
        W_final = max_x - min_x
        H_final = max_y - min_y
        if W_final > 0 and H_final > 0:
            final_cp = _contact_polish(result_2d, W_final, H_final, max_time=4.0)
            if _check_valid(final_cp.flatten(), W_final, H_final):
                result_2d = final_cp

    # Scale to ensure perimeter <= 4
    result_2d = _scale_to_perimeter(result_2d)

    # Final validation and cleanup
    xs = result_2d[:, 0]
    ys = result_2d[:, 1]
    rs = result_2d[:, 2]

    # Ensure circles are inside their bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    W = max_x - min_x
    H = max_y - min_y

    # Shift to origin
    xs = xs - min_x
    ys = ys - min_y

    # Fix any remaining overlaps by scaling down radii
    for _ in range(10):
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                if dist < rs[i] + rs[j] - 1e-9:
                    factor = dist / (rs[i] + rs[j]) * 0.99
                    rs[i] *= factor
                    rs[j] *= factor
                    rs[i] = max(0, min(rs[i], min(xs[i], W - xs[i], ys[i], H - ys[i])))
                    rs[j] = max(0, min(rs[j], min(xs[j], W - xs[j], ys[j], H - ys[j])))

    result = np.column_stack((xs, ys, rs))
    return result
