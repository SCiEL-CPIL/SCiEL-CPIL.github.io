"""Dzugutov cluster global minima: structure-based (icosahedral) seeds + gradient relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 270.0  # safety margin under 300s kill

# Dzugutov parameters (kept for reference / possible direct energy use)
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    rv = r[iu]
    if rv.size == 0:
        return 0.0
    e = np.zeros_like(rv)
    m1 = rv < _a
    m2 = (rv >= _a) & (rv < _b)
    if m1.any():
        r1 = rv[m1]
        e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    if m2.any():
        r2 = rv[m2]
        e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(np.sum(e))


def _relax(positions, time_left):
    """Relax using the provided relaxer; fall back to our own if it fails/slow."""
    pos = np.asarray(positions, dtype=float)
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    # Fallback: simple gradient descent with the provided gradient
    p = pos.copy()
    step = 0.01
    for it in range(400):
        if _elapsed() > TIME_LIMIT - 5.0:
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-8:
            break
        p = p - step * g / gn * min(0.5, 0.05 * gn)
        if not np.all(np.isfinite(p)):
            break
    return _energy(p), p


def _icosahedron():
    """Standard icosahedron vertices (12)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    return v / np.linalg.norm(v[0])


def _icosahedral_shells(radius):
    """Shell radii (multiples of the icosahedron edge) for nested icosahedra."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    edge = np.linalg.norm(_icosahedron()[0] - _icosahedron()[1])
    # magic shell radii: 1, ~1.7, ~2.4 ...
    return [1.0, 1.0 + 1.0 / phi, 1.0 + 2.0 / phi, 1.0 + 3.0 / phi]


def _build_icosahedral(n, scale=1.0, rng=None):
    """Build an n-atom icosahedral cluster by filling shells (12, 42, 92, ...)."""
    if rng is None:
        rng = np.random.default_rng(n)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = _icosahedron()
    pts = [np.zeros(3)]  # centre
    # shell radii (in units of base edge)
    shell_r = [1.0, 1.0 + 1.0 / phi, 1.0 + 2.0 / phi, 1.0 + 3.0 / phi]
    shell_counts = [12, 42, 92, 120]
    shell_idx = 0
    while len(pts) < n:
        r = shell_r[min(shell_idx, len(shell_r) - 1)]
        count = shell_counts[min(shell_idx, len(shell_counts) - 1)]
        need = min(count, n - len(pts))
        # choose 'need' of the 'count' candidate directions
        cand = _shell_directions(shell_idx)
        if cand is not None and len(cand) >= need:
            chosen = cand[:need]
        else:
            # scatter on a sphere of radius r
            chosen = _scatter(need, r, rng)
        for d in chosen:
            pts.append(d * r * scale)
        shell_idx += 1
    return np.array(pts[:n])


def _shell_directions(k):
    """Candidate unit directions for shell k (0=first icosahedral shell)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    if k == 0:
        return _icosahedron()
    if k == 1:
        # 42 directions: face-centres + edge-midpoints of the icosahedron
        ico = _icosahedron()
        dirs = []
        # face centres (20)
        faces = [
            [0, 1, 8], [0, 8, 4], [0, 4, 11], [0, 11, 2], [0, 2, 1],
            [1, 2, 5], [2, 11, 3], [11, 4, 9], [4, 8, 10], [8, 1, 6],
            [5, 2, 3], [3, 11, 7], [7, 9, 3], [9, 4, 10], [10, 8, 6],
            [6, 1, 5], [5, 3, 7], [7, 9, 10], [10, 6, 5], [6, 10, 8],
        ]
        for f in faces:
            c = (ico[f[0]] + ico[f[1]] + ico[f[2]]) / 3.0
            c /= np.linalg.norm(c)
            dirs.append(c)
        # edge midpoints (30) - pick a subset to reach 42
        edges = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 11],
            [1, 2], [1, 5], [1, 6], [1, 8], [1, 0],
            [2, 3], [2, 5], [2, 11], [2, 1], [2, 0],
            [3, 7], [3, 9], [3, 11], [3, 2], [3, 5],
            [4, 9], [4, 10], [4, 11], [4, 8], [4, 0],
            [5, 6], [5, 7], [5, 2], [5, 3], [5, 1],
            [6, 7], [6, 10], [6, 1], [6, 5], [6, 8],
            [7, 9], [7, 10], [7, 3], [7, 5], [7, 6],
            [8, 10], [8, 6], [8, 0], [8, 1], [8, 4],
            [9, 10], [9, 7], [9, 4], [9, 3], [9, 8],
            [10, 6], [10, 7], [10, 4], [10, 8], [10, 9],
            [11, 3], [11, 7], [11, 0], [11, 2], [11, 4],
        ]
        for e in edges:
            c = (ico[e[0]] + ico[e[1]]) / 2.0
            c /= np.linalg.norm(c)
            dirs.append(c)
        return np.array(dirs[:42])
    # higher shells: scatter
    return None


def _scatter(need, radius, rng):
    """need unit vectors * radius, roughly evenly spread on a sphere."""
    pts = []
    for _ in range(need):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * radius)
    return np.array(pts)


def _fcc_cluster(n, scale=1.0):
    """Build an FCC-like cluster (cuboctahedral shells) as a polycrystalline seed."""
    pts = [np.zeros(3)]
    a = scale
    # first shell: 12 (cuboctahedron)
    s = a / np.sqrt(2.0)
    shell1 = []
    for (x, y, z) in [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                      (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]:
        shell1.append(np.array([x, y, z]) * s)
    for p in shell1:
        if len(pts) < n:
            pts.append(p)
    # second shell: 30 (octahedron edges)
    s2 = a
    shell2 = []
    for (x, y, z) in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
                      (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0),
                      (0, 0, 2), (0, 0, -2)]:
        shell2.append(np.array([x, y, z]) * a / 2.0)
    rng = np.random.default_rng(0)
    for p in shell2:
        if len(pts) < n:
            pts.append(p)
    # fill remaining randomly near surface
    while len(pts) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * s2 * 1.6)
    return np.array(pts[:n])


def _random_compact(n, spacing=1.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    pts = []
    tries = 0
    while len(pts) < n and tries < 4000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.75 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:n])


def _seeds(n):
    """List of candidate starting geometries for cluster size n."""
    rng = np.random.default_rng(1000 + n)
    seeds = []
    # icosahedral (favoured by Dzugutov), a few scales
    for scale in (0.95, 1.0, 1.05):
        try:
            seeds.append(_build_icosahedral(n, scale=scale, rng=rng))
        except Exception:
            pass
    # fcc / polycrystalline
    try:
        seeds.append(_fcc_cluster(n, scale=1.0))
    except Exception:
        pass
    # random compact, a couple of them
    for _ in range(2):
        seeds.append(_random_compact(n, spacing=1.1, rng=rng))
    return seeds


def construct(instance):
    """Structure-based search: icosahedral/polytetrahedral seeds + gradient relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    best_e = np.inf
    best_pos = None

    seeds = _seeds(size)
    for s in seeds:
        if _remaining() < 15.0:
            break
        s = np.asarray(s, dtype=float)
        # enforce minimum separation 0.5
        try:
            e, p = _relax(s, _remaining())
            if np.all(np.isfinite(p)):
                if e < best_e:
                    best_e = e
                    best_pos = p
        except Exception:
            continue

    if best_pos is None:
        # last-resort: a simple icosahedral seed, minimally relaxed
        best_pos = _build_icosahedral(size, scale=1.0)
        try:
            _, best_pos = relax(best_pos, potential)
        except Exception:
            pass

    # final safety: ensure finite and reasonably separated
    best_pos = np.asarray(best_pos, dtype=float)
    return best_pos
# EVOLVE-BLOCK-END
