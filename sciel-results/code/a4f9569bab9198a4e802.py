"""Dzugutov 38-60 atom clusters: icosahedral-seed basin hopping with parallel kicks."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 292.0  # safety margin under 300s wall-clock limit


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _time_left():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Exact Dzugutov energy (vectorized)."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    e = np.zeros_like(r)
    m1 = r < a
    m2 = (r >= a) & (r < b)
    if np.any(m1):
        e[m1] = A * (r[m1] ** (-m) - B) * np.exp(c / (r[m1] - a))
    if np.any(m2):
        e[m2] = B * np.exp(d / (r[m2] - b))
    return float(np.sum(e))


def _gold():
    return (1.0 + 5.0 ** 0.5) / 2.0


def _icosahedron():
    g = _gold()
    return np.array([
        [0, 1, g], [0, 1, -g], [0, -1, g], [0, -1, -g],
        [1, g, 0], [1, -g, 0], [-1, g, 0], [-1, -g, 0],
        [g, 0, 1], [g, 0, -1], [-g, 0, 1], [-g, 0, -1],
    ], dtype=float)


def _rotate(v, theta, axis):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return v @ (np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K)


def _make_icosahedral_seed(n, rng):
    """Polytetrahedral seed: a core icosahedron plus concentric icosahedral shells."""
    core = _icosahedron()
    core = core / np.linalg.norm(core, axis=1, keepdims=True)
    core = _rotate(core, rng.uniform(0, 2 * np.pi), np.array([0, 0, 1.0]))
    points = [core * 1.0]
    placed = 12
    r = 1.9
    while placed < n:
        shell = _icosahedron()
        shell = shell / np.linalg.norm(shell, axis=1, keepdims=True)
        shell = _rotate(shell, rng.uniform(0, 2 * np.pi), np.array([0, 0, 1.0]))
        shell = _rotate(shell, rng.uniform(0, 2 * np.pi), np.array([1.0, 0, 0]))
        shell = shell * r
        take = min(12, n - placed)
        points.append(shell[:take])
        placed += take
        r += 1.05
    pts = np.vstack(points)
    pts = pts[:n]
    pts = pts - pts.mean(axis=0)
    return pts


def _make_random_seed(n, rng):
    """Random compact packing (fallback diversity)."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    guard = 0
    while len(pts) < n and guard < 4000:
        guard += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:  # force fill to guarantee n atoms
        pts.append(rng.uniform(-radius, radius, 3))
    return np.array(pts[:n])


def _kick(pos, scale, rng):
    p = pos + scale * rng.standard_normal(pos.shape)
    return p


def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    n = instance["n"]
    rng = np.random.default_rng(n)

    # Build a pool of diverse starting structures.
    pool = []
    for k in range(6):
        pool.append(_make_icosahedral_seed(n, rng))
    for k in range(3):
        pool.append(_make_random_seed(n, rng))

    kick_scales = [0.15, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3]

    best = None
    best_e = np.inf

    def worker(start, scale, k):
        p = _kick(start, scale, rng)
        try:
            e, pos = relax(p, "dzugutov", instance.get("rho", instance.get("kind")))
        except Exception:
            return None
        if e < -1e12:
            return None
        return (float(e), np.asarray(pos, dtype=float))

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = []
        # First wave: relax each seed with a moderate kick.
        for s in pool:
            if _time_left() < 5.0:
                break
            scale = kick_scales[2]
            futures.append(ex.submit(worker, s, scale, 0))
        for f in as_completed(futures):
            res = f.result()
            if res is not None and res[0] < best_e:
                best_e, best = res[0], res[1]

        # Basin hopping: kick the current best at diverse scales, in parallel.
        it = 0
        while _time_left() > 12.0 and best is not None:
            it += 1
            scales = [kick_scales[it % len(kick_scales)],
                      kick_scales[(it + 3) % len(kick_scales)],
                      kick_scales[(it + 5) % len(kick_scales)],
                      kick_scales[(it + 1) % len(kick_scales)]]
            futures = [ex.submit(worker, best, sc, i) for i, sc in enumerate(scales)]
            local_best = None
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    if local_best is None or res[0] < local_best[0]:
                        local_best = res
                if _time_left() < 12.0:
                    break
            if local_best is not None and local_best[0] < best_e:
                best_e, best = local_best[0], local_best[1]
            # Occasionally re-seed from the pool to escape a deep basin.
            if it % 8 == 0:
                s = pool[it % len(pool)]
                res = worker(s, kick_scales[4], it)
                if res is not None and res[0] < best_e:
                    best_e, best = res[0], res[1]

    if best is None:
        # Fallback: relax a single icosahedral seed directly.
        e, pos = relax(_make_icosahedral_seed(n, rng), "dzugutov", instance.get("rho", instance.get("kind")))
        return pos
    return best
# EVOLVE-BLOCK-END
