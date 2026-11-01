"""Suite task: 50 equal spheres in a unit cube.
Strategy: multi-start (grid + random) -> smooth repulsion gradient descent
(soft-min surrogate) -> Metropolis Monte-Carlo -> final polish.
Deadline-guarded to stay under the 120 s wall-clock limit."""
import time
import numpy as np

N = 50
DEADLINE = 115.0          # safety margin vs 120 s limit
_t0 = time.perf_counter()

def _time_left():
    return DEADLINE - (time.perf_counter() - _t0)

def _score(c):
    """Return (radius, valid)."""
    c = np.asarray(c)
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0, False
    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 0.0, False
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    r = min(0.5 - np.max(np.abs(c)), 0.5 * np.min(dist))
    if not np.isfinite(r) or r <= 0.0:
        return 0.0, False
    return float(r), True

def _energy(c, beta):
    """Smooth surrogate: -softmin of pairwise distances + wall penalty."""
    c = np.asarray(c)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    m = np.min(dist)
    softmin = -np.logaddexp.reduce(np.log(dist / m) - beta * (dist - m)) * m
    # wall penalty: push out of cube
    over = np.clip(np.abs(c) - 0.5, 0.0, None)
    wall = np.sum(over * over) * 1e4
    return -softmin + wall

def _grad(c, beta):
    """Gradient of _energy."""
    c = np.asarray(c)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    m = np.min(dist)
    a = np.log(dist / m) - beta * (dist - m)
    # softmax weights w_ij = exp(-a_ij)/sum
    amax = np.max(a, axis=1, keepdims=True)
    w = np.exp(-(a - amax))
    np.fill_diagonal(w, 0.0)
    wsum = w.sum(axis=1, keepdims=True)
    w = w / wsum
    # gradient of -softmin: softmin = m*sum w_ij log(dist_ij/m)
    # d(-softmin)/dc_i = -sum_j w_ij * (dist_ij/m) * d(dist_ij)/dc_i
    # d(dist_ij)/dc_i = (c_i - c_j)/dist_ij
    # => d(-softmin)/dc_i = -sum_j w_ij * (1/m) * (c_i - c_j)
    g = -np.sum(w * d, axis=1) / m
    # wall penalty gradient
    over = np.clip(np.abs(c) - 0.5, 0.0, None)
    g += 1e4 * 2.0 * over * np.sign(c)
    return g

def _project(c):
    return np.clip(c, -0.5, 0.5)

def _grid_init():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init(rng):
    return (rng.uniform(-0.5, 0.5, size=(N, 3)))

def _grad_descent(c, beta, max_iter, step0, min_step):
    c = c.copy()
    step = step0
    best_e = _energy(c, beta)
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        g = _grad(c, beta)
        c = _project(c - step * g)
        e = _energy(c, beta)
        if e < best_e:
            best_e = e
        else:
            step *= 0.7
        step = max(step, min_step)
        if np.max(np.abs(g)) < 1e-12:
            break
    return c

def _metropolis(c, beta, n_iter, T0, T1, rng, step):
    c = c.copy()
    r0, _ = _score(c)
    best_r = r0
    best_c = c.copy()
    for it in range(n_iter):
        if _time_left() < 8.0:
            break
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        idx = rng.integers(0, N)
        nc = c.copy()
        nc[idx] = _project(nc[idx] + step * rng.normal(size=3))
        # quick radius via local distances
        d = np.linalg.norm(nc - nc[idx], axis=1)
        d[idx] = np.inf
        r = min(0.5 - np.max(np.abs(nc[idx])), 0.5 * np.min(d),
                0.5 - np.max(np.abs(nc)))
        # energy-based accept (smooth)
        e0 = _energy(c, beta)
        e1 = _energy(nc, beta)
        if e1 < e0 or rng.random() < np.exp(-(e1 - e0) / max(T, 1e-9)):
            c = nc
            r1, _ = _score(c)
            if r1 > best_r:
                best_r = r1
                best_c = c.copy()
    return best_c, best_r

def construct():
    rng = np.random.default_rng(12345)
    best_c = None
    best_r = -1.0

    starts = [_grid_init()]
    for _ in range(12):
        starts.append(_random_init(rng))

    for s in starts:
        if _time_left() < 12.0:
            break
        c = _grad_descent(s, beta=200.0, max_iter=3000, step0=0.02, min_step=1e-5)
        # metropolis to escape local minima
        c, r = _metropolis(c, beta=200.0, n_iter=20000, T0=0.05, T1=0.001,
                           rng=rng, step=0.03)
        # polish
        c = _grad_descent(c, beta=500.0, max_iter=2000, step0=0.01, min_step=1e-6)
        r, valid = _score(c)
        if valid and r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = _grid_init()
    return best_c
