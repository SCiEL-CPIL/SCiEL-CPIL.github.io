"""Circle packing in perimeter-4 rectangle: Force-directed equilibrium + radial growth."""
import numpy as np
import time
import math

N = 21


def _is_valid(x, y, r, W, H, tol=1e-7):
    """Check if all circles are within bounds and non-overlapping."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    # Pairwise check using numpy broadcasting for speed
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_uniform_scale(x, y, r, W, H, time_budget, max_iter=60):
    """Find the largest s such that (x, y, r*s) is valid. Returns s."""
    lo = 1.0
    hi = 1.0
    # Exponential search for upper bound
    for _ in range(50):
        if time.time() > time_budget:
            break
        if _is_valid(x, y, r * hi, W, H):
            lo = hi
            hi *= 1.15
            if hi > 5.0:
                break
        else:
            break
    if hi <= 1.0:
        return 1.0
    # Binary search
    for _ in range(max_iter):
        if time.time() > time_budget:
            break
        mid = (lo + hi) / 2.0
        if _is_valid(x, y, r * mid, W, H):
            lo = mid
        else:
            hi = mid
    return lo


def _force_equilibrium(x, y, r, W, H, time_budget, n_steps=50000, dt=0.002, k_rep=8.0, k_wall=30.0, damping=0.95):
    """
    Force-directed relaxation: circles repel each other when overlapping,
    walls push back. Radii grow slowly. We seek equilibrium where net force ~ 0.
    """
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)
    r = r.copy().astype(np.float64)
    vx = np.zeros(N)
    vy = np.zeros(N)

    grow_rate = 0.00003  # slow uniform growth

    for step in range(n_steps):
        if time.time() > time_budget:
            break

        # Grow radii
        r += grow_rate
        r = np.maximum(r, 0.003)

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        for i in range(N):
            # Left wall
            pen = r[i] - x[i]
            if pen > 0:
                fx[i] += k_wall * pen
            # Right wall
            pen = x[i] + r[i] - W
            if pen > 0:
                fx[i] -= k_wall * pen
            # Bottom wall
            pen = r[i] - y[i]
            if pen > 0:
                fy[i] += k_wall * pen
            # Top wall
            pen = y[i] + r[i] - H
            if pen > 0:
                fy[i] -= k_wall * pen

        # Pairwise repulsion (only when overlapping)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-12
                rsum = r[i] + r[j]
                if d < rsum:
                    overlap = rsum - d
                    f = k_rep * overlap
                    ux = dx / d
                    uy = dy / d
                    fx[i] += ux * f
                    fy[i] += uy * f
                    fx[j] -= ux * f
                    fy[j] -= uy * f

        # Update velocities with damping
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt

        # Update positions
        x += vx
        y += vy

        # Hard clamp to walls
        for i in range(N):
            x[i] = min(max(x[i], r[i]), W - r[i])
            y[i] = min(max(y[i], r[i]), H - r[i])

    return x, y, r


def _radial_growth_search(x, y, r, W, H, time_budget, n_rounds=30):
    """
    Iteratively: grow all radii by binary search, then do a short force relaxation
    to restore feasibility, then grow again. This "pump" approach can find larger
    packings than single-shot growth.
    """
    x = x.copy().astype(np.float64)
    y = y.copy().astype(np.float64)
    r = r.copy().astype(np.float64)

    for rnd in range(n_rounds):
        if time.time() > time_budget:
            break

        # Grow radii
        s = _max_uniform_scale(x, y, r, W, H, time_budget, max_iter=40)
        if s <= 1.0 + 1e-9:
            break
        r *= s

        # Short force relaxation to fix any numerical drift
        sub_budget = time_budget - (time.time() - (time_budget - 10)) if False else time_budget
        x, y, r = _force_equilibrium(x, y, r, W, H, time_budget, n_steps=2000, dt=0.001, k_rep=5.0, k_wall=20.0)

    # Final grow
    s = _max_uniform_scale(x, y, r, W, H, time_budget, max_iter=50)
    r *= s

    return x, y, r


def _hex_init(W, H, n_rows, n_cols):
    """Hexagonal grid initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    spacing_x = W / n_cols
    spacing_y = H / n_rows
    r0 = min(spacing_x, spacing_y) * 0.45

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * spacing_x
            cy = (row + 0.5) * spacing_y
            if row % 2 == 1:
                cx += spacing_x * 0.25
            x[count] = min(max(cx, r0), W - r0)
            y[count] = min(max(cy, r0), H - r0)
            r[count] = r0
            count += 1

    while count < N:
        x[count] = W * 0.5
        y[count] = H * 0.5
        r[count] = 0.02
        count += 1

    return x, y, r


def _grid_init(W, H):
    """Simple grid initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    n_cols = int(math.ceil(math.sqrt(N * W / H)))
    n_rows = int(math.ceil(N / n_cols))

    spacing_x = W / n_cols
    spacing_y = H / n_rows
    r0 = min(spacing_x, spacing_y) * 0.45

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            x[count] = (col + 0.5) * spacing_x
            y[count] = (row + 0.5) * spacing_y
            r[count] = r0
            count += 1

    while count < N:
        x[count] = W * 0.5
        y[count] = H * 0.5
        r[count] = 0.02
        count += 1

    return x, y, r


def _random_init(W, H, rng):
    """Random non-overlapping-ish initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    # Start with small circles and grow
    r[:] = 0.02
    for i in range(N):
        x[i] = rng.uniform(r[i], W - r[i])
        y[i] = rng.uniform(r[i], H - r[i])

    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    best_H = 1.0

    rng = np.random.default_rng(42)

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.7, 1.45, 0.65, 1.55, 0.6, 1.65, 0.55, 1.75, 0.5]

    for W0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
        H0 = 2.0 - W0
        if H0 < 0.35 or W0 < 0.35:
            continue

        # Multiple initializations per aspect ratio
        inits = []

        # Hexagonal
        n_cols = max(2, int(round(math.sqrt(N * W0 / H0))))
        n_rows = max(2, int(math.ceil(N / n_cols)))
        inits.append(_hex_init(W0, H0, n_rows, n_cols))

        # Grid
        inits.append(_grid_init(W0, H0))

        # Random (2 variants)
        for _ in range(2):
            inits.append(_random_init(W0, H0, rng))

        for x0, y0, r0 in inits:
            if time.time() > deadline - 10:
                break

            # Phase 1: Force equilibrium with slow growth
            t_budget1 = deadline - (time.time() - start_time) - 20
            if t_budget1 < 5:
                break

            x1, y1, r1 = _force_equilibrium(x0, y0, r0, W0, H0, start_time + t_budget1,
                                            n_steps=30000, dt=0.002, k_rep=8.0, k_wall=30.0)

            # Phase 2: Radial growth pump
            t_budget2 = deadline - (time.time() - start_time) - 10
            if t_budget2 > 5:
                x2, y2, r2 = _radial_growth_search(x1, y1, r1, W0, H0, start_time + t_budget2, n_rounds=20)
            else:
                x2, y2, r2 = x1, y1, r1

            # Phase 3: Final short force relaxation + grow
            t_budget3 = deadline - (time.time() - start_time) - 5
            if t_budget3 > 3:
                x3, y3, r3 = _force_equilibrium(x2, y2, r2, W0, H0, start_time + t_budget3,
                                                n_steps=5000, dt=0.001, k_rep=5.0, k_wall=20.0)
                s = _max_uniform_scale(x3, y3, r3, W0, H0, start_time + t_budget3, max_iter=40)
                r3 *= s
                x3, y3 = x3, y3
            else:
                x3, y3, r3 = x2, y2, r2

            if _is_valid(x3, y3, r3, W0, H0):
                val = float(np.sum(r3))
                if val > best_value:
                    best_value = val
                    best_x = x3.copy()
                    best_y = y3.copy()
                    best_r = r3.copy()
                    best_W = W0
                    best_H = H0

    # If we have a result, do one more intensive pass
    if best_x is not None and time.time() < deadline - 8:
        t_budget = deadline - (time.time() - start_time) - 5
        if t_budget > 5:
            # More aggressive force relaxation
            x4, y4, r4 = _force_equilibrium(best_x, best_y, best_r, best_W, best_H,
                                            start_time + t_budget, n_steps=40000, dt=0.0015,
                                            k_rep=10.0, k_wall=35.0)
            s = _max_uniform_scale(x4, y4, r4, best_W, best_H, start_time + t_budget, max_iter=50)
            r4 *= s
            if _is_valid(x4, y4, r4, best_W, best_H):
                val = float(np.sum(r4))
                if val > best_value:
                    best_value = val
                    best_x = x4.copy()
                    best_y = y4.copy()
                    best_r = r4.copy()

    # Fallback
    if best_x is None:
        W = 1.0
        H = 1.0
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01))
        return np.array(cells[:N])

    # Safety shrink
    for _ in range(30):
        r_trial = best_r * (1 - 1e-9)
        if _is_valid(best_x, best_y, r_trial, best_W, best_H):
            best_r = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, best_r))
