"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 125  # safety margin under 150s kill


def _radius(pts):
    """Compute the common radius for a set of circle centres."""
    n = len(pts)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # distance to walls (square [-0.5, 0.5]^2)
    wall = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall = wall.min()
    return min(min_pair, 2.0 * min_wall) / 2.0


def _neg_radius(pts):
    return -_radius(pts)


def _hex_initial(n):
    """Generate a hexagonal lattice initial placement inside [-0.5,0.5]^2."""
    pts = []
    # hex spacing
    dx = 1.0 / (np.sqrt(3.0) * 14)  # roughly 14 columns
    dy = dx * np.sqrt(3.0) / 2.0
    x0, y0 = -0.5 + dx, -0.5 + dy
    row = 0
    while len(pts) < n:
        offset = (row % 2) * dx / 2.0
        x = x0 + offset
        while x < 0.5 - dx / 2:
            y = y0 + row * dy
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                pts.append([x, y])
            x += dx
            if len(pts) >= n:
                break
        row += 1
    return np.array(pts[:n])


def _grid_initial(n):
    """Square grid initial placement."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:n]


def _optimize(pts0, max_time=110.0):
    """Run L-BFGS-B optimization starting from pts0, with time limit."""
    start = time.time()
    flat = pts0.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    def fun(x):
        if time.time() - start > max_time:
            raise ValueError("timeout")
        return _neg_radius(x.reshape(N, 2))

    def jac(x):
        # Numerical gradient (central differences)
        if time.time() - start > max_time:
            raise ValueError("timeout")
        p = x.reshape(N, 2)
        f0 = _neg_radius(p)
        g = np.zeros_like(x)
        step = 1e-6
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += step
            xm = x.copy()
            xm[i] -= step
            g[i] = (_neg_radius(xp.reshape(N, 2)) - _neg_radius(xm.reshape(N, 2))) / (2 * step)
        return g

    # Try L-BFGS-B with analytical-ish gradient via finite differences is too slow.
    # Use a simpler approach: just use fun and let scipy compute gradient.
    result = None
    try:
        result = minimize(
            fun, flat, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < _neg_radius(flat):
            flat = result.x
    except (ValueError, Exception):
        pass

    # Check time
    elapsed = time.time() - start
    if elapsed < max_time - 10:
        # Try a second round of optimization from the improved point
        try:
            result2 = minimize(
                fun, flat, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < _neg_radius(flat):
                flat = result2.x
        except (ValueError, Exception):
            pass

    return flat.reshape(N, 2)


def construct():
    t0 = time.time()
    # Try multiple initializations and pick the best
    best_pts = None
    best_r = -1.0

    initializers = [
        _hex_initial(N),
        _grid_initial(N),
    ]

    # Add a few perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _hex_initial(N).copy()
        noise = rng.normal(0, 0.01, size=base.shape)
        base = np.clip(base + noise, -0.49, 0.49)
        initializers.append(base)

    for init in initializers:
        if time.time() - t0 > 115:
            break
        try:
            pts = _optimize(init, max_time=min(110.0, 115.0 - (time.time() - t0)))
            r = _radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts
        except Exception:
            continue

    if best_pts is None:
        best_pts = _grid_initial(N)

    # Final time check
    if time.time() - t0 > 130:
        return best_pts

    return best_pts
# EVOLVE-BLOCK-END
