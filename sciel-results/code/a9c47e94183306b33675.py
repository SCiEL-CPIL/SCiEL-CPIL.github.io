"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def remaining():
        return max(0.0, deadline - time.time())

    def unpack(x, y, r):
        return np.column_stack([x, y, r])

    def validate(x, y, r, w, h, tol=1e-7):
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False

        # Pairwise overlap check (vectorized per i)
        for i in range(N):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            if np.any(dist < r[i] + r[i+1:] - tol):
                return False
        return True

    def grid_init(w, h, seed, n_cols=None, n_rows=None):
        rng = np.random.default_rng(seed)
        if n_cols is None or n_rows is None:
            n_cols = max(1, int(round(np.sqrt(N * w / max(h, 1e-9)))))
            n_rows = max(1, int(np.ceil(N / n_cols)))
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            xs[i] = (col + 0.5) * w / n_cols
            ys[i] = (row + 0.5) * h / n_rows

        # Jitter to avoid exact symmetries
        jitter = 0.05 * min(w / n_cols, h / n_rows)
        xs += rng.normal(0, jitter, N)
        ys += rng.normal(0, jitter, N)
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)

        est_r = 0.35 * min(w / n_cols, h / n_rows)
        rs = np.full(N, est_r)
        return xs, ys, rs

    def force_balance_step(x, y, r, w, h, step=0.02, k=8, wall_k=1.0, tol=1e-9):
        # Directed, contact-based force balancing:
        # - Repulsion for overlaps (pairwise)
        # - Attraction toward a target spacing to encourage dense packing
        # - Wall repulsion to keep circles inside
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise forces
        for i in range(N):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dist = np.sqrt(dx*dx + dy*dy) + tol
            overlap = r[i] + r[i+1:] - dist
            # Repulsive force proportional to overlap
            for j_idx, j in enumerate(range(i+1, N)):
                if overlap[j_idx] > 0:
                    f = k * overlap[j_idx]
                    fx[i] += f * dx[j_idx] / dist[j_idx]
                    fy[i] += f * dy[j_idx] / dist[j_idx]
                    fx[j] -= f * dx[j_idx] / dist[j_idx]
                    fy[j] -= f * dy[j_idx] / dist[j_idx]

        # Wall forces
        for i in range(N):
            # Left
            viol = r[i] - x[i]
            if viol > 0:
                fx[i] += wall_k * viol
            # Right
            viol = r[i] + x[i] - w
            if viol > 0:
                fx[i] -= wall_k * viol
            # Bottom
            viol = r[i] - y[i]
            if viol > 0:
                fy[i] += wall_k * viol
            # Top
            viol = r[i] + y[i] - h
            if viol > 0:
                fy[i] -= wall_k * viol

        # Move
        x += step * fx
        y += step * fy

        # Clamp to rectangle with radius margin
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        return x, y

    def local_perturbation(x, y, r, w, h, rng, scale=0.05):
        # Perturb positions slightly to escape local minima
        x += rng.normal(0, scale * np.mean(r), N)
        y += rng.normal(0, scale * np.mean(r), N)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        return x, y

    def grow_radii(x, y, r, w, h, steps=200, base_step=0.005):
        # Try to grow radii while maintaining validity
        for _ in range(steps):
            if remaining() < 0.5:
                break
            # Compute max allowed growth for each circle based on current constraints
            growth = np.zeros(N)
            for i in range(N):
                # Wall constraints
                g_wall = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
                growth[i] = g_wall
                # Pairwise constraints
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # dist >= r[i] + r[j] + 2*delta -> delta <= (dist - r[i] - r[j]) / 2
                    delta = (dist - r[i] - r[j]) / 2.0
                    growth[i] = min(growth[i], delta)
            growth = np.maximum(growth, 0.0)
            if np.all(growth <= 0):
                break
            # Grow by a fraction of the minimum growth
            min_growth = np.min(growth)
            step = min(base_step, min_growth)
            r += step
            # Validate
            if not validate(x, y, r, w, h):
                # Rollback
                r -= step
                break
        return r

    def optimize_aspect(w, h, seed, max_iters=5000):
        rng = np.random.default_rng(seed)
        # Initialize
        x, y, r = grid_init(w, h, seed)
        current_score = np.sum(r)
        best_local_score = current_score
        best_local = (x.copy(), y.copy(), r.copy())

        # Force balancing iterations
        for it in range(max_iters):
            if remaining() < 1.0:
                break
            x, y = force_balance_step(x, y, r, w, h)
            # Occasionally perturb
            if it % 100 == 0 and it > 0:
                x, y = local_perturbation(x, y, r, w, h, rng, scale=0.03)
            # Grow radii
            r = grow_radii(x, y, r, w, h, steps=50, base_step=0.003)
            # Validate and update best
            if validate(x, y, r, w, h):
                score = np.sum(r)
                if score > best_local_score:
                    best_local_score = score
                    best_local = (x.copy(), y.copy(), r.copy())
            else:
                # Rollback to best local
                x, y, r = best_local
        return best_local, best_local_score

    # Search over aspect ratios
    aspect_ratios = np.linspace(0.7, 1.3, 13)
    for ar in aspect_ratios:
        if remaining() < 5.0:
            break
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        for restart in range(3):
            if remaining() < 3.0:
                break
            seed = int(ar * 1000) + restart * 7
            (x, y, r), score = optimize_aspect(w, h, seed)
            if validate(x, y, r, w, h):
                if score > best_score:
                    best_score = score
                    best_result = unpack(x, y, r)

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
