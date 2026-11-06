"""Dzugutov cluster global minima: improved basin hopping with better seeds and annealing."""
import time
import numpy as np
import concurrent.futures
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 290.0
_STOP = threading.Event()

_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, TIME_LIMIT - _elapsed())


def _energy(positions):
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
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


def _relax(positions, deadline=None):
    pos = np.asarray(positions, dtype=float)
    try:
        energy, out = relax(pos, "dzugutov")
        if out is not None:
            out = np.asarray(out, dtype=float)
            if out.shape == pos.shape and np.all(np.isfinite(out)):
                return float(energy), out
    except Exception:
        pass
    p = pos.copy()
    best_e = _energy(p)
    best_p = p.copy()
    lr = 0.05
    for it in range(600):
        if deadline is not None and _elapsed() > deadline:
            break
        if _STOP.is_set():
            break
        try:
            g = dzugutov_energy_gradient(p)
        except Exception:
            break
        g = np.asarray(g, dtype=float)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        new_p = p - lr * g / gn
        if not np.all(np.isfinite(new_p)):
            lr *= 0.5
            new_p = p - lr * g / gn
            if not np.all(np.isfinite(new_p)):
                break
        e_new = _energy(new_p)
        if e_new < best_e:
            best_e = e_new
            best_p = new_p.copy()
            p = new_p
            lr = min(lr * 1.05, 0.3)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    return best_e, best_p


def _icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    edge = np.linalg.norm(v[0] - v[1])
    return v / edge


def _icosahedral_shells():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    return [
        (0.0, 1),
        (1.0, 12),
        (1.0 + 1.0 / phi, 42),
        (1.0 + 2.0 / phi, 92),
        (1.0 + 3.0 / phi, 120),
    ]


def _build_icosahedral(n, scale=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(n)
    shells = _icosahedral_shells()
    pts = []
    for radius, count in shells:
        if len(pts) >= n:
            break
        need = min(count, n - len(pts))
        if radius == 0.0:
            pts.append(np.zeros(3) * scale)
        else:
            shell_num = int(round(radius - 1.0)) if radius > 1.0 else 1
            cand = _shell_directions(shell_num)
            if cand is not None and len(cand) >= need:
                chosen = cand[:need]
            else:
                chosen = _scatter(need, 1.0, rng)
            for d in chosen:
                pts.append(d * radius * scale)
    while len(pts) < n:
        outer_r = shells[-1][0] * scale
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v * outer_r * 1.05)
    return np.array(pts[:n])


def _shell_directions(shell_num):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = _icosahedron()

    if shell_num == 1:
        return ico

    if shell_num == 2:
        dirs = []
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
        edges = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 11],
            [1, 2], [1, 5], [1, 6], [1, 8],
            [2, 3], [2, 5], [2, 11],
            [3, 7], [3, 9], [3, 11],
            [4, 9], [4, 10], [4, 11],
            [5, 6], [5, 7],
            [6, 7], [6, 10],
        ]
        for e in edges:
            c = (ico[e[0]] + ico[e[1]]) / 2.0
            c /= np.linalg.norm(c)
            dirs.append(c)
        return np.array(dirs[:42])

    n_pts = 92 if shell_num == 3 else 120
    rng = np.random.default_rng(shell_num * 1000)
    pts = []
    for i in range(n_pts):
        z = 1 - 2 * (i + 0.5) / n_pts
        r = np.sqrt(1 - z * z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _scatter(need, radius, rng):
    pts = []
    for i in range(need):
        z = 1 - 2 * (i + 0.5) / need
        r = np.sqrt(1 - z * z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        v = np.array([r * np.cos(theta), r * np.sin(theta), z])
        pts.append(v * radius)
    return np.array(pts)


def _fcc_cluster(n, scale=1.0):
    pts = [np.zeros(3)]
    a = scale
    s = a / np.sqrt(2.0)
    shell1 = []
    for (x, y, z) in [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                      (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]:
        shell1.append(np.array([x, y, z]) * s)
    for p in shell1:
        if len(pts) < n:
            pts.append(p)
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
    rng = np.random.default_rng(1000 + n)
    seeds = []
    try:
        seeds.append(_build_icosahedral(n, scale=1.0, rng=rng))
    except Exception:
        pass
    for scale in (0.90, 0.95, 1.05, 1.10):
        try:
            seeds.append(_build_icosahedral(n, scale=scale, rng=rng))
        except Exception:
            pass
    try:
        seeds.append(_fcc_cluster(n, scale=1.0))
    except Exception:
        pass
    for spacing in (1.0, 1.15, 1.3):
        seeds.append(_random_compact(n, spacing=spacing, rng=rng))
    if seeds:
        base = seeds[0]
        for _ in range(2):
            perturbed = base + rng.normal(0, 0.15, size=base.shape)
            seeds.append(perturbed)
    return seeds


def _basin_hopping_worker(seed_pos, n, deadline, worker_id):
    pos = np.asarray(seed_pos, dtype=float).copy()
    best_e = np.inf
    best_pos = pos.copy()

    try:
        e, p = _relax(pos, deadline)
        if np.all(np.isfinite(p)):
            best_e = e
            best_pos = p.copy()
    except Exception:
        best_e = _energy(pos)
        best_pos = pos.copy()

    if _STOP.is_set() or _elapsed() > deadline:
        return best_e, best_pos

    amp = 0.20
    temp = 0.8
    rng = np.random.default_rng(42 + worker_id * 1000 + n)
    current_e = best_e
    current_pos = best_pos.copy()

    while not _STOP.is_set() and _elapsed() < deadline:
        noise = rng.normal(0, amp, size=current_pos.shape)
        perturbed = current_pos + noise
        if not np.all(np.isfinite(perturbed)):
            amp *= 0.5
            continue

        try:
            e_new, p_new = _relax(perturbed, deadline)
        except Exception:
            e_new = _energy(perturbed)
            p_new = perturbed

        if not np.all(np.isfinite(p_new)):
            amp *= 0.5
            continue

        if e_new < current_e:
            current_e = e_new
            current_pos = p_new.copy()
            amp = min(amp * 1.05, 0.5)
        else:
            delta_e = e_new - current_e
            if delta_e > 0:
                p_accept = np.exp(-delta_e / max(temp, 1e-8))
                if rng.random() > p_accept:
                    amp = max(amp * 0.92, 0.05)
                    temp = max(temp * 0.97, 0.05)
                    continue
            current_e = e_new
            current_pos = p_new.copy()

        if e_new < best_e:
            best_e = e_new
            best_pos = p_new.copy()

        temp = max(temp * 0.99, 0.03)

    return best_e, best_pos


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]

    seeds = _seeds(size)

    n_workers = min(4, len(seeds))

    deadline = _elapsed() + max(10.0, _remaining() - 10.0)

    results = []
    best_e = np.inf
    best_pos = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            seed = seeds[i % len(seeds)]
            f = executor.submit(_basin_hopping_worker, seed, size, deadline, i)
            futures.append(f)

        for f in futures:
            try:
                e, p = f.result(timeout=max(5.0, _remaining()))
                if np.all(np.isfinite(p)):
                    results.append((e, p))
                    if e < best_e:
                        best_e = e
                        best_pos = p.copy()
            except Exception:
                continue

    if best_pos is None:
        for s in seeds:
            try:
                e, p = _relax(s, _elapsed() + max(5.0, _remaining()))
                if np.all(np.isfinite(p)) and e < best_e:
                    best_e = e
                    best_pos = p.copy()
            except Exception:
                continue

    if best_pos is None:
        best_pos = _build_icosahedral(size, scale=1.0)
        try:
            _, best_pos = relax(best_pos, potential)
        except Exception:
            pass

    best_pos = np.asarray(best_pos, dtype=float)
    return best_pos
