"""Logarithmic energy minimization of 100 points on the unit sphere."""
import time
import numpy as np

N = 100
# Pair index matrix (i<j) for vectorized energy/gradient.
_I, _J = np.triu_indices(N, k=1)
_NPAIRS = _I.shape[0]


def _energy(pts):
    d = np.linalg.norm(pts[_I] - pts[_J], axis=1)
    return float(np.sum(np.log(1.0 / d)))


def _grad(pts):
    diff = pts[_I] - pts[_J]
    d2 = np.einsum('ij,ij->i', diff, diff)
    d = np.sqrt(d2)
    # d/dp_i log(1/|p_i-p_j|) = -(p_i - p_j)/|p_i-p_j|^2
    w = diff / d2
    g = np.zeros_like(pts)
    np.add.at(g, _I, -w)
    np.add.at(g, _J, w)
    return g


def _project(pts):
    n = np.linalg.norm(pts, axis=1)
    n = np.where(n > 0, n, 1.0)
    return pts / n[:, None]


def _valid(pts):
    if pts is None or not np.all(np.isfinite(pts)):
        return False
    n = np.linalg.norm(pts, axis=1)
    if not np.all(n > 1e-12):
        return False
    d = np.linalg.norm(pts[_I] - pts[_J], axis=1)
    return bool(np.all(d > 1e-9))


def _fibonacci(k):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _ring(k):
    m = 10
    i = np.arange(N)
    col = i // m
    row = i % m
    z = 1.0 - 2.0 * (col + 0.5) / m
    theta = 2.0 * np.pi * row / m + (col % 2) * (np.pi / m)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _rand_unit(rng):
    v = rng.standard_normal((N, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v


def _rot(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _make_seeds():
    rng = np.random.default_rng(12345)
    seeds = []
    seeds.append(_fibonacci(0))
    seeds.append(_ring(0))
    for _ in range(8):
        seeds.append(_rand_unit(rng))
    for _ in range(12):
        seeds.append(_rot(rng) @ _fibonacci(0))
    for _ in range(8):
        seeds.append(_rot(rng) @ _ring(0))
    return seeds


def construct():
    import scipy.optimize as so

    deadline = time.time() + 112.0
    seeds = _make_seeds()

    best_pts = _fibonacci(0)
    best_e = _energy(best_pts)

    for s in seeds:
        if time.time() > deadline:
            break
        p0 = _project(s.copy())
        if not _valid(p0):
            continue
        try:
            res = so.minimize(
                _energy, p0, jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxfun': 3000,
                         'gtol': 1e-12, 'ftol': 1e-15},
            )
        except Exception:
            continue
        pts = _project(np.asarray(res.x, dtype=float))
        if not _valid(pts):
            continue
        e = _energy(pts)
        if np.isfinite(e) and e < best_e:
            best_e = e
            best_pts = pts

    # Final safety: return a guaranteed-valid, unit-normalized configuration.
    out = _project(best_pts)
    if not _valid(out):
        out = _project(_fibonacci(0))
    return out
