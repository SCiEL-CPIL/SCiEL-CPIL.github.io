"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def get_rect_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def solve_radii_lp(x, y, w, h):
        """Solve LP: max sum r_i subject to non-overlap and boundary constraints.
        Returns (radii, objective) or (None, -inf) if infeasible."""
        from scipy.optimize import linprog
        c = -np.ones(N)  # minimize -sum r_i
        A_ub = []
        b_ub = []
        # Non-overlap: r_i + r_j <= dist_ij
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                A_ub.append(np.zeros(N))
                A_ub[-1][i] = 1.0
                A_ub[-1][j] = 1.0
                b_ub.append(d)
        # Boundary: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        for i in range(N):
            for bound_val in [x[i], w - x[i], y[i], h - y[i]]:
                row = np.zeros(N)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(max(bound_val, 0.0))
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(1e-6, None)] * N,
                      method='highs')
        if res.status != 0:
            return None, -np.inf
        radii = res.x
        # Clip tiny negatives
        radii = np.maximum(radii, 1e-6)
        return radii, -res.fun

    def check_valid(x, y, radii, w, h, eps=1e-7):
        if np.any(radii < 1e-6):
            return False
        if np.any(x - radii < -eps) or np.any(x + radii > w + eps):
            return False
        if np.any(y - radii < -eps) or np.any(y + radii > h + eps):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < radii[i] + radii[j] - eps:
                    return False
        return True

    def make_grid_positions(t, rows, cols, offset_rows=0, offset_cols=0):
        """Generate grid positions for N circles in rectangle with parameter t."""
        w, h = get_rect_dims(t)
        total_cells = rows * cols
        # Fill cells in a specific order
        positions = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                cx = (c + 0.5) / cols * w
                cy = (r + 0.5) / rows * h
                positions.append((cx, cy))
                idx += 1
        return np.array(positions)

    def evaluate_packing(x, y, t):
        w, h = get_rect_dims(t)
        radii, obj = solve_radii_lp(x, y, w, h)
        if radii is None:
            return None, -np.inf
        if not check_valid(x, y, radii, w, h):
            return None, -np.inf
        return radii, obj

    def perturb_positions(x, y, w, h, scale):
        """Randomly perturb positions within bounds."""
        new_x = x + random.gauss(0, scale) * np.ones(N)
        new_y = y + random.gauss(0, scale) * np.ones(N)
        # Keep within rectangle
        new_x = np.clip(new_x, 0.01, w - 0.01)
        new_y = np.clip(new_y, 0.01, h - 0.01)
        return new_x, new_y

    def local_search(x, y, t, time_budget):
        """Iteratively perturb positions and re-solve LP."""
        w, h = get_rect_dims(t)
        best_x, best_y, best_r, best_obj = x.copy(), y.copy(), None, -np.inf
        best_x, best_y, best_r, best_obj = x.copy(), y.copy(), None, -np.inf
        radii, obj = solve_radii_lp(best_x, best_y, w, h)
        if radii is not None and check_valid(best_x, best_y, radii, w, h):
            best_r, best_obj = radii, obj

        scale = 0.05 * min(w, h)
        no_improve = 0
        while time.time() < time_budget:
            new_x, new_y = perturb_positions(best_x, best_y, w, h, scale)
            new_r, new_obj = evaluate_packing(new_x, new_y, t)
            if new_r is not None and new_obj > best_obj:
                best_x, best_y, best_r, best_obj = new_x, new_y, new_r, new_obj
                no_improve = 0
                scale = max(0.005 * min(w, h), scale * 0.95)
            else:
                no_improve += 1
                scale = max(0.002 * min(w, h), scale * 0.98)
            if no_improve > 50:
                break
        return best_x, best_y, best_r, best_obj

    def optimize_aspect(x, y, radii, t, time_budget):
        """Try different aspect ratios, re-solve LP for each."""
        best_t = t
        best_r = radii
        best_obj = np.sum(radii)
        t_vals = np.linspace(0.15, 0.85, 20)
        for ti in t_vals:
            if time.time() > time_budget:
                break
            w, h = get_rect_dims(ti)
            # Scale positions to fit new rectangle
            old_w, old_h = get_rect_dims(t)
            sx = w / old_w
            sy = h / old_h
            new_x = x * sx
            new_y = y * sy
            new_r, obj = solve_radii_lp(new_x, new_y, w, h)
            if new_r is not None and check_valid(new_x, new_y, new_r, w, h):
                if obj > best_obj:
                    best_t, best_r, best_obj = ti, new_r, obj
        return best_t, best_r, best_obj

    # Main algorithm
    best_result = None
    best_score = -np.inf

    # Grid configurations to try: (rows, cols)
    grid_configs = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4), (3, 6), (6, 3)]

    for restart in range(12):
        if time.time() > deadline - 20:
            break

        if restart < len(grid_configs):
            rows, cols = grid_configs[restart]
        else:
            rows, cols = random.choice(grid_configs)

        # Random initial aspect ratio
        t_init = random.uniform(0.35, 0.65)
        w, h = get_rect_dims(t_init)

        # Generate positions
        x, y = make_grid_positions(t_init, rows, cols)
        if len(x) < N:
            # Pad with random positions
            extra = N - len(x)
            extra_x = np.random.uniform(0.1, w - 0.1, extra)
            extra_y = np.random.uniform(0.1, h - 0.1, extra)
            x = np.concatenate([x, extra_x])
            y = np.concatenate([y, extra_y])

        # Solve LP for radii
        radii, obj = solve_radii_lp(x, y, w, h)
        if radii is None:
            continue
        if not check_valid(x, y, radii, w, h):
            continue

        # Local search on positions
        time_for_ls = min(8, deadline - 25 - time.time())
        if time_for_ls < 2:
            break
        x, y, radii, obj = local_search(x, y, t_init, time.time() + time_for_ls)
        if radii is None:
            continue

        # Optimize aspect ratio
        time_for_aspect = min(10, deadline - 20 - time.time())
        if time_for_aspect < 2:
            break
        t_init, radii, obj = optimize_aspect(x, y, radii, t_init, time.time() + time_for_aspect)

        if obj > best_score:
            best_score = obj
            best_result = (x, y, radii, t_init)

    if best_result is None:
        # Fallback: simple valid grid
        t = 0.5
        w, h = get_rect_dims(t)
        x, y = make_grid_positions(t, 5, 5)
        if len(x) < N:
            extra = N - len(x)
            x = np.concatenate([x, np.random.uniform(0.1, w - 0.1, extra)])
            y = np.concatenate([y, np.random.uniform(0.1, h - 0.1, extra)])
        radii, _ = solve_radii_lp(x, y, w, h)
        if radii is None:
            radii = np.full(N, 0.01)
        best_result = (x, y, radii, t)

    x, y, radii, t = best_result

    # Final validity check
    w, h = get_rect_dims(t)
    if not check_valid(x, y, radii, w, h):
        # Shrink radii until valid
        for _ in range(100):
            if check_valid(x, y, radii, w, h):
                break
            radii *= 0.98
        if not check_valid(x, y, radii, w, h):
            # Last resort
            radii = np.full(N, 0.01)
            x = np.linspace(0.05, w - 0.05, N)
            y = np.full(N, h / 2)

    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = radii
    return result

# EVOLVE-BLOCK-END
