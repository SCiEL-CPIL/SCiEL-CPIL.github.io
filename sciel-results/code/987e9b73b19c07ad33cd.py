"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def check_time():
        return time.time() > deadline

    # --- Geometry helpers ---
    def compute_rect(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def evaluate_equal(circles, w, h):
        """Evaluate a packing with equal radii. Returns (r, valid)."""
        rs = circles[:, 2]
        # Check if radii are roughly equal
        r_mean = np.mean(rs)
        if np.max(rs) - np.min(rs) > 0.01:
            return 0.0, False

        r = r_mean
        if r <= 0:
            return 0.0, False

        xs = circles[:, 0]
        ys = circles[:, 1]

        # Boundary check
        if np.any(xs < r - 1e-9) or np.any(xs > w - r + 1e-9) or \
           np.any(ys < r - 1e-9) or np.any(ys > h - r + 1e-9):
            return 0.0, False

        # Pairwise distance check
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 2 * r - 1e-9:
                    return 0.0, False

        return r, True

    def evaluate_general(circles, aspect):
        """General evaluation for varying radii. Returns (sum_radii, valid)."""
        w, h = compute_rect(aspect)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return 0.0, False

        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return 0.0, False

        return float(np.sum(rs)), True

    # --- Initialization strategies ---
    def init_hexagonal(rows, aspect):
        """Create hexagonal packing with given row counts."""
        if sum(rows) != N:
            return None

        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))

        if len(pts) != N:
            return None

        pts = np.array(pts)
        center = pts.mean(axis=0)
        pts -= center

        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()

        w_total = w_centers + 2.0
        h_total = h_centers + 2.0

        # Scale to fit aspect ratio
        # We want w/h = aspect
        # Current aspect is w_total/h_total
        # Scale x and y differently? No, uniform scale first, then adjust aspect via optimization
        # For initialization, just scale to fit a box with given aspect
        # Let's scale so that it fits in w x h where w+h=2 and w/h=aspect
        w_rect = 2.0 * aspect / (1.0 + aspect)
        h_rect = 2.0 / (1.0 + aspect)

        # Uniform scale to fit
        scale_x = w_rect / w_total
        scale_y = h_rect / h_total
        scale = min(scale_x, scale_y)

        xs_final = pts[:, 0] * scale + w_rect / 2.0
        ys_final = pts[:, 1] * scale + h_rect / 2.0
        r = scale

        circles = np.zeros((N, 3))
        circles[:, 0] = xs_final
        circles[:, 1] = ys_final
        circles[:, 2] = r

        return circles, aspect

    def init_grid(rows, cols, aspect):
        """Create grid packing."""
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break

        if len(pts) < N:
            return None

        pts = np.array(pts[:N])
        center = pts.mean(axis=0)
        pts -= center

        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()

        w_total = w_centers + 2.0
        h_total = h_centers + 2.0

        w_rect = 2.0 * aspect / (1.0 + aspect)
        h_rect = 2.0 / (1.0 + aspect)

        scale_x = w_rect / w_total
        scale_y = h_rect / h_total
        scale = min(scale_x, scale_y)

        xs_final = pts[:, 0] * scale + w_rect / 2.0
        ys_final = pts[:, 1] * scale + h_rect / 2.0
        r = scale

        circles = np.zeros((N, 3))
        circles[:, 0] = xs_final
        circles[:, 1] = ys_final
        circles[:, 2] = r

        return circles, aspect

    def init_random(aspect):
        """Random initialization."""
        w, h = compute_rect(aspect)
        circles = np.zeros((N, 3))
        circles[:, 0] = np.random.uniform(0.1, w - 0.1, N)
        circles[:, 1] = np.random.uniform(0.1, h - 0.1, N)
        circles[:, 2] = 0.05 + 0.05 * np.random.rand(N)
        return circles, aspect

    # --- Optimization routines ---
    def optimize_equal_radius(circles, aspect, max_iter=200):
        """
        Optimize equal-radius packing using SLSQP.
        Variables: w, r, xs, ys. h = 2 - w.
        Objective: maximize r (minimize -r).
        """
        if check_time():
            return circles, aspect, 0.0

        w, h = compute_rect(aspect)
        r_init = np.mean(circles[:, 2])

        v0 = np.zeros(2 + 2 * N)
        v0[0] = w
        v0[1] = r_init
        v0[2:2 + N] = circles[:, 0]
        v0[2 + N:2 + 2 * N] = circles[:, 1]

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2 + N]
            ys = v[2 + N:2 + 2 * N]

            cons = []

            # Aspect sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})

            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})

            # Overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: ((v[2 + i] - v[2 + j]) ** 2 +
                                                    (v[2 + N + i] - v[2 + N + j]) ** 2) -
                                                    4 * (v[1] ** 2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2 * N)

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})

            if res.success or res.fun < -r_init:
                w_res = res.x[0]
                r_res = res.x[1]
                xs_res = res.x[2:2 + N]
                ys_res = res.x[2 + N:2 + 2 * N]

                # Validate
                valid = True
                h_res = 2.0 - w_res
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                dist_sq = (xs_res[i] - xs_res[j]) ** 2 + (ys_res[i] - ys_res[j]) ** 2
                                if dist_sq < 4 * (r_res ** 2) - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break

                if valid:
                    new_circles = np.zeros((N, 3))
                    new_circles[:, 0] = xs_res
                    new_circles[:, 1] = ys_res
                    new_circles[:, 2] = r_res
                    new_aspect = w_res / (2.0 - w_res)
                    score = r_res * N
                    return new_circles, new_aspect, score

        except Exception:
            pass

        return circles, aspect, 0.0

    def polish_general(circles, aspect, max_iter=100):
        """
        Polish general packing (varying radii) using SLSQP.
        Variables: w, xs, ys, rs. h = 2 - w.
        Objective: maximize sum(rs).
        """
        if check_time():
            return circles, aspect, 0.0

        w, h = compute_rect(aspect)

        v0 = np.zeros(1 + 3 * N)
        v0[0] = w
        v0[1:1 + N] = circles[:, 0]
        v0[1 + N:1 + 2 * N] = circles[:, 1]
        v0[1 + 2 * N:1 + 3 * N] = circles[:, 2]

        def objective(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            rs = v[1 + 2 * N:1 + 3 * N]
            if w_val <= 0.1 or h_val <= 0.1 or np.any(rs <= 0):
                return 1e6
            return -np.sum(rs)

        def constraints(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            xs = v[1:1 + N]
            ys = v[1 + N:1 + 2 * N]
            rs = v[1 + 2 * N:1 + 3 * N]

            cons = []

            # Aspect sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})

            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1 + i] - v[1 + 2 * N + i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1 + i] - v[1 + 2 * N + i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1 + N + i] - v[1 + 2 * N + i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[1 + N + i] - v[1 + 2 * N + i]})

            # Overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: ((v[1 + i] - v[1 + j]) ** 2 +
                                                    (v[1 + N + i] - v[1 + N + j]) ** 2) -
                                                    (v[1 + 2 * N + i] + v[1 + 2 * N + j]) ** 2
                    })
            return cons

        bounds = [(0.2, 1.8)] + [(0.0, 2.0)] * (2 * N) + [(0.01, 1.0)] * N

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})

            if res.success:
                w_res = res.x[0]
                xs_res = res.x[1:1 + N]
                ys_res = res.x[1 + N:1 + 2 * N]
                rs_res = res.x[1 + 2 * N:1 + 3 * N]

                # Validate
                valid = True
                h_res = 2.0 - w_res
                if w_res <= 0 or h_res <= 0 or np.any(rs_res <= 0):
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < rs_res[i] - 1e-7 or xs_res[i] > w_res - rs_res[i] + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < rs_res[i] - 1e-7 or ys_res[i] > h_res - rs_res[i] + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                dist_sq = (xs_res[i] - xs_res[j]) ** 2 + (ys_res[i] - ys_res[j]) ** 2
                                if dist_sq < (rs_res[i] + rs_res[j]) ** 2 - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break

                if valid:
                    new_circles = np.zeros((N, 3))
                    new_circles[:, 0] = xs_res
                    new_circles[:, 1] = ys_res
                    new_circles[:, 2] = rs_res
                    new_aspect = w_res / (2.0 - w_res)
                    score = float(np.sum(rs_res))
                    return new_circles, new_aspect, score

        except Exception:
            pass

        return circles, aspect, 0.0

    # --- Main search loop ---
    best_score = -1.0
    best_result = None

    random.seed(42)
    np.random.seed(42)

    # List of aspects to try
    aspects = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 0.4, 0.75, 1.33, 2.5]

    # Hexagonal row configurations
    hex_configs = [
        [7, 7, 7],
        [4, 5, 4, 4, 4],
        [3, 4, 4, 4, 4, 2],
        [5, 5, 5, 3, 3],
        [6, 6, 5, 4],
        [2, 4, 4, 4, 4, 3],
        [5, 4, 4, 4, 4],
    ]

    # Grid configurations
    grid_configs = [
        (5, 5),
        (3, 7),
        (4, 6),
        (7, 3),
        (6, 4),
        (7, 4),
        (4, 5),
    ]

    # Generate initial configurations
    initial_configs = []

    # Add hexagonal inits for a few aspects
    for aspect in [0.8, 1.0, 1.25]:
        for rows in hex_configs[:3]:
            cfg = init_hexagonal(rows, aspect)
            if cfg is not None:
                initial_configs.append(cfg)

    # Add grid inits
    for aspect in [0.8, 1.0, 1.25]:
        for rows, cols in grid_configs[:3]:
            cfg = init_grid(rows, cols, aspect)
            if cfg is not None:
                initial_configs.append(cfg)

    # Add random inits
    for aspect in [0.5, 0.8, 1.0, 1.5, 2.0]:
        for _ in range(3):
            cfg = init_random(aspect)
            if cfg is not None:
                initial_configs.append(cfg)

    # Process each initial configuration
    for circles, aspect in initial_configs:
        if check_time():
            break

        # First, try equal-radius optimization
        new_circles, new_aspect, score = optimize_equal_radius(circles, aspect, max_iter=150)
        if score > best_score:
            best_score = score
            best_result = (new_circles, new_aspect)

        # Then, try general polish
        if not check_time():
            new_circles2, new_aspect2, score2 = polish_general(circles, aspect, max_iter=100)
            if score2 > best_score:
                best_score = score2
                best_result = (new_circles2, new_aspect2)

    # If no result found, create a fallback
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w, h = compute_rect(1.0)
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            circles[i, 2] = 0.5 / side * min(w, h) * 0.95
        best_result = (circles, 1.0)
        best_score = 0.0

    # Final polish on the best result
    if not check_time():
        final_circles, final_aspect = best_result[0].copy(), best_result[1]
        final_circles, final_aspect, final_score = polish_general(final_circles, final_aspect, max_iter=200)
        if final_score > best_score:
            best_score = final_score
            best_result = (final_circles, final_aspect)

    # Clamp radii to ensure validity
    circles, aspect = best_result
    w, h = compute_rect(aspect)

    for i in range(N):
        r = min(circles[i, 2], circles[i, 0] - 1e-6, w - circles[i, 0] - 1e-6,
                circles[i, 1] - 1e-6, h - circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        circles[i, 2] = r

    return circles

# EVOLVE-BLOCK-END
