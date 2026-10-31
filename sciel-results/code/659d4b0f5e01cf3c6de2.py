"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural rewrite:
  * The perimeter is that of the TIGHT bounding box of the circles.  For a fixed
    perimeter the square maximizes area, and the objective (sum of radii) is
    area-like, so we optimise in a fixed unit square [0,1]^2 (perimeter 4) and
    simply verify the tight box has perimeter <= 4.
  * Vectorised penalty objective (no Python inner loops) with a growing penalty.
  * Several diverse initialisations (square grid, hexagonal, random, and a
    hand-designed 21-circle layout) to escape local minima.
  * SLSQP on the penalty, then a vectorised local search that pushes each circle
    to enlarge its radius, then a final tight-bound recompute.
"""
import numpy as np
import time

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 135.0


def _time_ok():
    return time.time() < DEADLINE


def _tight_box(circles):
    """Tight bounding box of the circles. Returns (bw, bh, perimeter)."""
    minx = np.min(circles[:, 0] - circles[:, 2])
    miny = np.min(circles[:, 1] - circles[:, 2])
    maxx = np.max(circles[:, 0] + circles[:, 2])
    maxy = np.max(circles[:, 1] + circles[:, 2])
    bw = maxx - minx
    bh = maxy - miny
    return bw, bh, 2.0 * (bw + bh)


def _validity(circles, tol=1e-7):
    """Check disjoint interiors and tight-box perimeter. Returns (ok, per, sum_r)."""
    if len(circles) != N:
        return False, 0.0, 0.0
    bw, bh, per = _tight_box(circles)
    if per > PERIMETER + 1e-6:
        return False, per, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(circles[i, 0] - circles[j, 0], circles[i, 1] - circles[j, 1])
            if d < circles[i, 2] + circles[j, 2] - tol:
                return False, per, 0.0
    return True, per, float(np.sum(circles[:, 2]))


def _penalty(circles, mu):
    """Vectorised penalty: -sum(r) + mu*(wall + overlap violations)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    val = -np.sum(r)
    # wall violations (circle must stay inside [0,1]^2)
    val += mu * np.sum(np.maximum(0.0, r - x) ** 2)
    val += mu * np.sum(np.maximum(0.0, r - y) ** 2)
    val += mu * np.sum(np.maximum(0.0, r + x - 1.0) ** 2)
    val += mu * np.sum(np.maximum(0.0, r + y - 1.0) ** 2)
    # overlap violations
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    min_d = r[:, None] + r[None, :]
    overlap = np.maximum(0.0, min_d - d)
    val += mu * np.sum(overlap ** 2)
    return val


def _max_radius(circles, i):
    """Largest radius circle i could have at its current position."""
    x = circles[i, 0]
    y = circles[i, 1]
    nr = min(x, y, 1.0 - x, 1.0 - y)
    for j in range(N):
        if j == i:
            continue
        d = np.hypot(x - circles[j, 0], y - circles[j, 1])
        cand = d - circles[j, 2]
        if cand < nr:
            nr = cand
    return max(nr, 0.0)


def _local_search(circles, iters, rng):
    """Vectorised local search: nudge a circle to a position that grows its radius."""
    circles = circles.copy()
    for it in range(iters):
        if not _time_ok():
            break
        i = rng.randrange(N)
        x, y = circles[i, 0], circles[i, 1]
        cur = _max_radius(circles, i)
        best_r = cur
        best_x, best_y = x, y
        # try a ring of candidate directions at a few radii
        for rad in (0.004, 0.010, 0.020, 0.035):
            for k in range(8):
                ang = 2.0 * np.pi * k / 8.0 + rng.uniform(-0.2, 0.2)
                nx = min(max(x + rad * np.cos(ang), 0.0), 1.0)
                ny = min(max(y + rad * np.sin(ang), 0.0), 1.0)
                nr = min(nx, ny, 1.0 - nx, 1.0 - ny)
                for j in range(N):
                    if j == i:
                        continue
                    d = np.hypot(nx - circles[j, 0], ny - circles[j, 1])
                    cand = d - circles[j, 2]
                    if cand < nr:
                        nr = cand
                nr = max(nr, 0.0)
                if nr > best_r + 1e-12:
                    best_r, best_x, best_y = nr, nx, ny
        if best_r > cur + 1e-12:
            circles[i, 0] = best_x
            circles[i, 1] = best_y
            circles[i, 2] = best_r
    return circles


def _normalize(circles):
    """Shift so the tight box sits at the origin; radii unchanged."""
    minx = np.min(circles[:, 0] - circles[:, 2])
    miny = np.min(circles[:, 1] - circles[:, 2])
    circles = circles.copy()
    circles[:, 0] -= minx
    circles[:, 1] -= miny
    return circles


def _slsqp_solve(circles0, mu):
    """Run SLSQP on the vectorised penalty starting from circles0."""
    from scipy.optimize import minimize
    x0 = circles0.reshape(-1)
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((1e-4, 0.5))
    obj = lambda v: _penalty(v.reshape(N, 3), mu)
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-10})
        if res.fun < obj(x0):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return circles0


def _init_grid(cols):
    rows = int(np.ceil(N / cols))
    cw = 1.0 / cols
    ch = 1.0 / rows
    cx = []
    cy = []
    for i in range(cols):
        for j in range(rows):
            if len(cx) >= N:
                break
            cx.append((i + 0.5) * cw)
            cy.append((j + 0.5) * ch)
    r = np.full(N, min(cw, ch) * 0.42)
    return np.column_stack([np.array(cx), np.array(cy), r])


def _init_hex(rng):
    """Hexagonal packing of equal circles, clipped to the square."""
    pts = []
    s = 0.16
    row = 0
    while len(pts) < N:
        y = s * (row + 0.5)
        xoff = (s * 0.866) * (row % 2)
        x = s * 0.5 + xoff
        while x < 1.0 and len(pts) < N:
            pts.append((x, y))
            x += 2 * s * 0.866
        row += 1
    cx = np.array([p[0] for p in pts])
    cy = np.array([p[1] for p in pts])
    r = np.full(N, s * 0.9)
    return np.column_stack([cx, cy, r])


def _init_random(rng):
    cx = rng.uniform(0.05, 0.95, N)
    cy = rng.uniform(0.05, 0.95, N)
    r = np.full(N, 0.05)
    return np.column_stack([cx, cy, r])


def _init_hand():
    """A hand-designed 21-circle layout (roughly 4-5-6-5-5 by row)."""
    data = [
        # row 1 (4)
        (0.13, 0.13), (0.37, 0.13), (0.63, 0.13), (0.87, 0.13),
        # row 2 (5)
        (0.10, 0.33), (0.30, 0.33), (0.50, 0.33), (0.70, 0.33), (0.90, 0.33),
        # row 3 (6)
        (0.08, 0.50), (0.25, 0.50), (0.42, 0.50), (0.58, 0.50), (0.75, 0.50), (0.92, 0.50),
        # row 4 (5)
        (0.10, 0.67), (0.30, 0.67), (0.50, 0.67), (0.70, 0.67), (0.90, 0.67),
        # row 5 (5)
        (0.10, 0.87), (0.30, 0.87), (0.50, 0.87), (0.70, 0.87), (0.90, 0.87),
    ]
    cx = np.array([p[0] for p in data])
    cy = np.array([p[1] for p in data])
    r = np.full(N, 0.09)
    return np.column_stack([cx, cy, r])


def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 135.0
    rng = np.random.default_rng(12345)

    best = None
    best_val = -1.0

    def _consider(circles):
        nonlocal best, best_val
        circles = np.clip(circles, 0.0, 1.0)
        circles = _normalize(circles)
        # set each radius to its maximum at its position
        for i in range(N):
            circles[i, 2] = _max_radius(circles, i)
        ok, per, s = _validity(circles)
        if ok and s > best_val:
            best_val = s
            best = circles.copy()

    # Diverse initialisations, optimised with growing penalty.
    inits = [
        _init_grid(5),
        _init_grid(4),
        _init_hex(rng),
        _init_hand(),
        _init_random(rng),
        _init_random(rng),
    ]

    for k, c0 in enumerate(inits):
        if not _time_ok():
            break
        circles = c0.copy()
        # A couple of penalty-growth SLSQP passes.
        for mu in (20.0, 100.0, 500.0):
            if not _time_ok():
                break
            circles = _slsqp_solve(circles, mu)
        # Local search to enlarge radii.
        steps = 800 if k < 4 else 400
        circles = _local_search(circles, steps, np.random.default_rng(1000 + k))
        _consider(circles)

    # If we have budget, do a couple more random-restart local searches from the
    # current best, perturbing positions slightly.
    if best is not None and _time_ok():
        for attempt in range(4):
            if not _time_ok():
                break
            pert = best.copy()
            pert[:, 0] = np.clip(pert[:, 0] + rng.normal(0, 0.02, N), 0.0, 1.0)
            pert[:, 1] = np.clip(pert[:, 1] + rng.normal(0, 0.02, N), 0.0, 1.0)
            pert[:, 2] = np.full(N, 0.05)
            circles = _slsqp_solve(pert, 100.0)
            circles = _local_search(circles, 500, np.random.default_rng(2000 + attempt))
            _consider(circles)

    if best is None:
        # Fallback: simple grid, normalised.
        best = _init_grid(5)
        for i in range(N):
            best[i, 2] = _max_radius(best, i)

    return np.array(best)


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    ok, per, s = _validity(res)
    print("valid:", ok, "perimeter:", per, "sum_r:", s)
