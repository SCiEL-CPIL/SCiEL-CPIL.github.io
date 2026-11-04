"""
Circle packing in unit square.
New strategy family: stochastic global search (simulated annealing) on the exact
nonsmooth objective sum(radii), plus repulsion-relaxed initialisation and an exact
local polish. This replaces the deterministic soft-min / L-BFGS-B gradient flow.
"""
import numpy as np
import time

N = 26
DEADLINE = 140.0  # safety margin from 150 s wall-clock limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Exact radii from centers: min of half-neighbor-distance and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def value(x, y):
        r = compute_radii(x, y)
        return float(np.sum(r)), r

    def check_validity(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        need = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(need, 0)
        if np.any(dist < need - tol):
            return False
        return True

    def repulsion_relax(x0, y0, iters=250, rng=None):
        """Push overlapping / poorly-spread circles apart so the local optimizer
        starts from a geometrically sound, well-distributed configuration."""
        x = x0.copy()
        y = y0.copy()
        for _ in range(iters):
            if time.time() - start_time > DEADLINE - 20:
                break
            # For each circle, find nearest neighbor and nudge away; also push from walls.
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            nn = np.argmin(dist, axis=1)
            # repulsion strength proportional to overlap proxy
            for i in range(N):
                j = nn[i]
                d = dist[i, j]
                if d < 1e-9:
                    if rng is not None:
                        x[i] += rng.uniform(-0.01, 0.01)
                        y[i] += rng.uniform(-0.01, 0.01)
                    continue
                # desired separation ~ 0.5 (two unit-ish radii); push if closer
                desired = 0.5
                if d < desired:
                    push = (desired - d) * 0.5 / d
                    x[i] += push * (x[i] - x[j])
                    y[i] += push * (y[i] - y[j])
            # wall repulsion
            for i in range(N):
                if x[i] < 0.05:
                    x[i] += (0.05 - x[i]) * 0.5
                elif x[i] > 0.95:
                    x[i] -= (x[i] - 0.95) * 0.5
                if y[i] < 0.05:
                    y[i] += (0.05 - y[i]) * 0.5
                elif y[i] > 0.95:
                    y[i] -= (y[i] - 0.95) * 0.5
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
        return x, y

    def hex_init(seed):
        rng = np.random.RandomState(seed)
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for row in range(6):
            for col in range(5):
                if idx >= N:
                    break
                x[idx] = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
                y[idx] = (row + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def grid_init(seed):
        rng = np.random.RandomState(seed)
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) / 5.0
                y[idx] = (i + 0.5) / 6.0
                idx += 1
                if idx >= N:
                    break
        x += rng.uniform(-0.04, 0.04, N)
        y += rng.uniform(-0.04, 0.04, N)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        return x, y

    def random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y

    def best_fit_local_search(x, y, max_rounds=120):
        """Exact discrete local search on sum(radii)."""
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [-1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)]
        ])
        x = x.copy()
        y = y.copy()
        _, r = value(x, y)
        curr_val = float(np.sum(r))

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            best_improvement = 1e-12
            best_move = None
            for step in step_sizes:
                for i in range(N):
                    for d in range(8):
                        nx = x[i] + dirs[d, 0] * step
                        ny = y[i] + dirs[d, 1] * step
                        if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                            continue
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = nx
                        y_temp[i] = ny
                        nv, _ = value(x_temp, y_temp)
                        improvement = nv - curr_val
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (i, d, step)
                if best_improvement > step * 0.05:
                    break
            if best_move is not None:
                i, d, step = best_move
                x[i] += dirs[d, 0] * step
                y[i] += dirs[d, 1] * step
                _, r = value(x, y)
                curr_val = float(np.sum(r))
            else:
                break
        _, r = value(x, y)
        return x, y, r, float(np.sum(r))

    def simulated_annealing(x0, y0, rng, t0=0.05, t1=1e-4, steps=4000):
        """SA on exact sum(radii). Move = pick a circle, Gaussian nudge."""
        x = x0.copy()
        y = y0.copy()
        curr_val, _ = value(x, y)
        best_val = curr_val
        best_x = x.copy()
        best_y = y.copy()
        sigma = 0.03

        for s in range(steps):
            if time.time() - start_time > DEADLINE - 8:
                break
            T = t0 * (t1 / t0) ** (s / max(1, steps - 1))
            i = rng.randint(N)
            dx = rng.normal(0, sigma)
            dy = rng.normal(0, sigma)
            nx = x[i] + dx
            ny = y[i] + dy
            if nx < 0.005 or nx > 0.995 or ny < 0.005 or ny > 0.995:
                continue
            x_temp = x.copy()
            y_temp = y.copy()
            x_temp[i] = nx
            y_temp[i] = ny
            nv, _ = value(x_temp, y_temp)
            delta = nv - curr_val
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
                x = x_temp
                y = y_temp
                curr_val = nv
                if curr_val > best_val:
                    best_val = curr_val
                    best_x = x.copy()
                    best_y = y.copy()
            # slowly shrink noise
            sigma = max(0.005, sigma * 0.9995)
        return best_x, best_y, best_val

    # ---- Multi-start orchestration ----
    best_val = -np.inf
    best_x = None
    best_y = None
    best_r = None

    rng = np.random.RandomState(12345)
    inits = [
        (hex_init, 0), (grid_init, 1), (random_init, 2),
        (hex_init, 10), (grid_init, 11), (random_init, 12),
        (hex_init, 20), (grid_init, 21), (random_init, 22),
    ]

    for init_func, seed in inits:
        if time.time() - start_time > DEADLINE - 25:
            break
        x0, y0 = init_func(seed)
        # repulsion relaxation for a well-distributed start
        x0, y0 = repulsion_relax(x0, y0, iters=150, rng=rng)
        v0, r0 = value(x0, y0)

        # Simulated annealing (stochastic global search)
        if time.time() - start_time < DEADLINE - 15:
            xs, ys, vs = simulated_annealing(x0, y0, rng, t0=0.04, t1=1e-4, steps=3000)
            if vs > v0:
                x0, y0, v0 = xs, ys, vs
                r0 = compute_radii(x0, y0)

        # Exact local polish
        if time.time() - start_time < DEADLINE - 10:
            xb, yb, rb, vb = best_fit_local_search(x0, y0, max_rounds=100)
            if vb > v0:
                x0, y0, r0, v0 = xb, yb, rb, vb

        if v0 > best_val:
            best_val = v0
            best_x = x0.copy()
            best_y = y0.copy()
            best_r = r0.copy()

    # Final high-precision polish on the best configuration
    if best_x is not None and time.time() - start_time < DEADLINE - 5:
        xb, yb, rb, vb = best_fit_local_search(best_x, best_y, max_rounds=200)
        if vb > best_val:
            best_val = vb
            best_x, best_y, best_r = xb, yb, rb

    # Ensure validity; shrink radii if numerically needed
    if best_x is None:
        x, y = hex_init(0)
        best_r = compute_radii(x, y)
        best_x, best_y = x, y
    elif not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
