"""Circle packing in perimeter-4 rectangle: Simulated Annealing with adaptive moves."""
import numpy as np
import time
import math

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_w = None

    def compute_value(x, y, r, w):
        return float(np.sum(r))

    def is_valid(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        if np.any(x - r < -tol) or np.any(x + r > w + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > h + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                rs = r[i] + r[j]
                if d2 < rs * rs - tol:
                    return False
        return True

    def get_initial_guesses():
        guesses = []
        rng = np.random.default_rng(12345)

        # Grid-based initializations with various aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.2, 0.4]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.25:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)

            # Try different grid arrangements
            for cols in [4, 5, 3]:
                rows = math.ceil(N / cols)
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_base = min(cell_w, cell_h) * 0.42

                for i in range(N):
                    row, col = divmod(i, cols)
                    if row >= rows:
                        break
                    x[i] = (col + 0.5) * cell_w
                    y[i] = (row + 0.5) * cell_h
                    r[i] = r_base

                if is_valid(x, y, r, w0):
                    guesses.append((x.copy(), y.copy(), r.copy(), w0))

        # Random scatter initializations
        for _ in range(15):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            if h0 < 0.25:
                continue
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            r_base = min(w0, h0) * 0.08
            for i in range(N):
                x[i] = rng.uniform(r_base, w0 - r_base)
                y[i] = rng.uniform(r_base, h0 - r_base)
                r[i] = r_base * rng.uniform(0.8, 1.2)
            guesses.append((x.copy(), y.copy(), r.copy(), w0))

        return guesses

    def simulated_annealing(x0, y0, r0, w0, time_budget):
        """Run simulated annealing to maximize sum of radii."""
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        r = r0.copy().astype(float)
        w = float(w0)

        current_value = compute_value(x, y, r, w)
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        best_w = w
        best_val = current_value

        t_start = time.time()
        T = 0.15
        cooling = 0.9995
        min_T = 1e-6

        step_pos = 0.02
        step_r = 0.005
        step_w = 0.02

        while T > min_T:
            if time.time() > time_budget:
                break

            # Choose move type
            move_type = np.random.choice(['pos', 'radius', 'width', 'jitter'], p=[0.4, 0.3, 0.15, 0.15])

            if move_type == 'pos':
                i = np.random.randint(N)
                dx = np.random.uniform(-step_pos, step_pos)
                dy = np.random.uniform(-step_pos, step_pos)
                new_x = x[i] + dx
                new_y = y[i] + dy
                h = 2.0 - w
                new_x = np.clip(new_x, r[i], w - r[i])
                new_y = np.clip(new_y, r[i], h - r[i])

                # Check validity of this move
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dxj = new_x - x[j]
                    dyj = new_y - y[j]
                    if dxj * dxj + dyj * dyj < (r[i] + r[j]) ** 2 - 1e-9:
                        valid = False
                        break

                if valid:
                    old_x, old_y = x[i], y[i]
                    x[i], y[i] = new_x, new_y
                    new_val = current_value  # radii unchanged
                    if new_val > current_value or np.random.rand() < math.exp((new_val - current_value) / T):
                        current_value = new_val
                        if current_value > best_val:
                            best_val = current_value
                            best_x = x.copy()
                            best_y = y.copy()
                            best_r = r.copy()
                            best_w = w
                    else:
                        x[i], y[i] = old_x, old_y

            elif move_type == 'radius':
                i = np.random.randint(N)
                delta_r = np.random.uniform(-step_r, step_r)
                new_r = r[i] + delta_r
                h = 2.0 - w

                if new_r < 0.005:
                    new_r = 0.005
                # Check wall constraints
                if new_r > x[i] or new_r > w - x[i] or new_r > y[i] or new_r > h - y[i]:
                    new_r = min(x[i], w - x[i], y[i], h - y[i]) * 0.999

                # Check overlap with others
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    rs = new_r + r[j]
                    if d2 < rs * rs - 1e-9:
                        valid = False
                        break

                if valid:
                    old_r = r[i]
                    r[i] = new_r
                    new_val = current_value + (new_r - old_r)
                    if new_val > current_value or np.random.rand() < math.exp((new_val - current_value) / T):
                        current_value = new_val
                        if current_value > best_val:
                            best_val = current_value
                            best_x = x.copy()
                            best_y = y.copy()
                            best_r = r.copy()
                            best_w = w
                    else:
                        r[i] = old_r

            elif move_type == 'width':
                delta_w = np.random.uniform(-step_w, step_w)
                new_w = w + delta_w
                if new_w < 0.3 or new_w > 1.7:
                    continue
                new_h = 2.0 - new_w

                # Check if all circles fit in new rectangle
                valid = True
                for i in range(N):
                    if x[i] - r[i] < -1e-9 or x[i] + r[i] > new_w + 1e-9:
                        valid = False
                        break
                    if y[i] - r[i] < -1e-9 or y[i] + r[i] > new_h + 1e-9:
                        valid = False
                        break

                if valid:
                    old_w = w
                    w = new_w
                    new_val = current_value  # radii unchanged
                    if new_val > current_value or np.random.rand() < math.exp((new_val - current_value) / T):
                        current_value = new_val
                        if current_value > best_val:
                            best_val = current_value
                            best_x = x.copy()
                            best_y = y.copy()
                            best_r = r.copy()
                            best_w = w
                    else:
                        w = old_w

            elif move_type == 'jitter':
                # Small random jitter to all positions
                for i in range(N):
                    dx = np.random.uniform(-step_pos * 0.3, step_pos * 0.3)
                    dy = np.random.uniform(-step_pos * 0.3, step_pos * 0.3)
                    x[i] = np.clip(x[i] + dx, r[i], w - r[i])
                    y[i] = np.clip(y[i] + dy, r[i], 2.0 - w - r[i])

                if is_valid(x, y, r, w):
                    new_val = current_value
                    if new_val > current_value or np.random.rand() < math.exp((new_val - current_value) / T):
                        current_value = new_val
                        if current_value > best_val:
                            best_val = current_value
                            best_x = x.copy()
                            best_y = y.copy()
                            best_r = r.copy()
                            best_w = w

            T *= cooling

            # Adaptive step sizes
            if T < 0.01:
                step_pos *= 0.95
                step_r *= 0.95
                step_w *= 0.95

        return best_x, best_y, best_r, best_w, best_val

    def local_polish(x, y, r, w, time_budget):
        """Local improvement: try to grow radii and adjust positions."""
        h = 2.0 - w
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(N):
                # Try to grow radius i
                r_wall = min(x[i], w - x[i], y[i], h - y[i])
                r_max = r_wall
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = math.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
                r_max = max(0.0, r_max)

                if r_max > r[i] + 1e-7:
                    r[i] = r_max * 0.9999
                    improved = True

                # Try to move circle i to a better position
                best_pos_val = 0.0
                best_px, best_py = x[i], y[i]
                for _ in range(20):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dist = np.random.uniform(0.001, 0.05)
                    nx = x[i] + dist * math.cos(angle)
                    ny = y[i] + dist * math.sin(angle)
                    nx = np.clip(nx, r[i], w - r[i])
                    ny = np.clip(ny, r[i], h - r[i])

                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break

                    if valid:
                        # Estimate potential improvement
                        min_gap = float('inf')
                        for j in range(N):
                            if i == j:
                                continue
                            dx = nx - x[j]
                            dy = ny - y[j]
                            d = math.sqrt(dx * dx + dy * dy)
                            gap = d - (r[i] + r[j])
                            if gap < min_gap:
                                min_gap = gap
                        wall_gap = min(nx - r[i], w - nx - r[i], ny - r[i], h - ny - r[i])
                        total_gap = min_gap + wall_gap
                        if total_gap > best_pos_val:
                            best_pos_val = total_gap
                            best_px, best_py = nx, ny

                if best_px != x[i] or best_py != y[i]:
                    x[i], y[i] = best_px, best_py
                    improved = True

        return x, y, r, w

    # Main loop: try multiple initializations and run SA on each
    guesses = get_initial_guesses()
    rng = np.random.default_rng(999)

    # Shuffle guesses
    order = rng.permutation(len(guesses))

    for idx in order:
        if time.time() > deadline - 30:
            break

        x0, y0, r0, w0 = guesses[idx]
        if not is_valid(x0, y0, r0, w0):
            continue

        time_budget = deadline - 5
        x_sa, y_sa, r_sa, w_sa, val_sa = simulated_annealing(x0, y0, r0, w0, time_budget)

        if val_sa > best_value:
            best_value = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()
            best_w = w_sa

    # Final polish on best result
    if best_x is not None and time.time() < deadline - 10:
        x_p, y_p, r_p, w_p = local_polish(best_x, best_y, best_r, best_w, deadline - 2)
        if is_valid(x_p, y_p, r_p, w_p):
            val_p = compute_value(x_p, y_p, r_p, w_p)
            if val_p > best_value:
                best_value = val_p
                best_x = x_p
                best_y = y_p
                best_r = r_p
                best_w = w_p

    # Safety: ensure validity with tiny shrink
    if best_x is not None:
        h = 2.0 - best_w
        for _ in range(20):
            r_trial = best_r * (1 - 1e-10)
            if is_valid(best_x, best_y, r_trial, best_w):
                best_r = r_trial
            else:
                break

    if best_x is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.01
            cells.append((cx, cy, r0))
        return np.array(cells)

    return np.column_stack((best_x, best_y, best_r))
