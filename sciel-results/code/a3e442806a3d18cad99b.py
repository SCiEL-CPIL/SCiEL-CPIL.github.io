"""Sutton-Chen cluster global-minima search: diverse structural prototypes + basin-hopping refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 280.0


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _fcc(n):
    pts = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            for k in range(-20, 21):
                pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _bcc(n):
    pts = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            for k in range(-20, 21):
                pts.append([i, j, k])
                pts.append([(i + 0.5), (j + 0.5), (k + 0.5)])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _hcp(n):
    pts = []
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    for layer in range(-15, 16):
        z = layer * h
        if layer % 3 == 0:
            off = (0.0, 0.0)
        elif layer % 3 == 1:
            off = (a / 2.0, a / (2 * np.sqrt(3.0)))
        else:
            off = (0.0, a / np.sqrt(3.0))
        for i in range(-15, 16):
            for j in range(-15, 16):
                x = i * a + (a / 2.0 if j % 2 else 0.0) + off[0]
                y = j * a * np.sqrt(3.0) / 2.0 + off[1]
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _icosahedral(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v1 = np.array([0.0, 1.0, t])
    v2 = np.array([1.0, t, 0.0])
    v3 = np.array([t, 0.0, 1.0])
    verts = []
    for v in (v1, v2, v3):
        for s in (1, -1):
            verts.append(v * s)
            verts.append(v[[1, 0, 2]] * s)
            verts.append(v[[2, 1, 0]] * s)
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1)[:, None]

    verts2 = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:
                verts2.append((verts[i] + verts[j]) / 2.0)
    used = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                a, b, c = verts[i], verts[j], verts[k]
                da = np.linalg.norm(a - b)
                db = np.linalg.norm(b - c)
                dc = np.linalg.norm(c - a)
                if da < 0.5 and db < 0.5 and dc < 0.5:
                    fc = (a + b + c) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    key = tuple(np.round(fc, 3))
                    if key not in used:
                        used.add(key)
                        verts2.append(fc)
    verts2 = np.array(verts2, dtype=float)

    verts3 = verts * np.sqrt(5.0)

    all_pts = np.vstack([verts, verts2, verts3])
    r = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(r)[:n]
    return all_pts[idx]


def _random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.55 * spacing
    attempts = 0
    max_attempts = 30000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _sphere_shell(n, rng, spacing=0.72):
    points = []
    shells = int(np.ceil(n ** (1.0 / 3.0)))
    for s in range(1, shells + 1):
        shell_radius = spacing * s * 0.8
        k = np.pi * (5 ** 0.5)
        i = len(points)
        while len(points) < n and i < n + s:
            t = (i + 0.5) / n
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            phi = k * i
            p = shell_radius * np.array([r * np.cos(phi), r * np.sin(phi), z])
            points.append(p)
            i += 1
    if len(points) > n:
        points = points[:n]
    while len(points) < n:
        p = rng.uniform(-spacing, spacing, 3)
        points.append(p)
    return np.array(points)


def _perturb(positions, rng, scale):
    return positions + rng.normal(0, scale, positions.shape)


def _relax_soft_hard(pos, potential, kind):
    e0, p0 = sutton_chen_energy_gradient(pos, kind)
    if not _valid(p0):
        return np.inf, pos
    e1, p1 = relax(pos, potential, kind)
    if _valid(p1) and e1 < e0:
        return e1, p1
    return e0, p0


def _basin_hop(start_pos, potential, kind, rng, max_iters=30, base_scale=0.25):
    current_pos = start_pos.copy()
    current_e, _ = sutton_chen_energy_gradient(current_pos, kind)
    if not _valid(current_pos):
        return current_e, current_pos
    current_e, current_pos = relax(current_pos, potential, kind)
    if not _valid(current_pos):
        return np.inf, start_pos

    best_e = current_e
    best_pos = current_pos.copy()
    temp = 0.4

    for it in range(max_iters):
        if _remaining() < 10.0:
            break
        scale = base_scale * (1.0 + 0.3 * np.sin(it * 0.4))
        perturbed = _perturb(current_pos, rng, scale)
        if not _valid(perturbed):
            continue
        try:
            e, p = relax(perturbed, potential, kind)
        except Exception:
            continue
        if not _valid(p):
            continue

        delta_e = e - current_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / (temp + 1e-10)):
            current_pos = p
            current_e = e
            if e < best_e:
                best_e = e
                best_pos = p.copy()
                temp *= 0.85
        else:
            temp *= 1.05
        temp = max(0.1, min(2.0, temp))

    return best_e, best_pos


def construct(instance):
    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    rng = np.random.default_rng(n * 7 + 13)

    prototypes = {
        "fcc": _fcc,
        "hcp": _hcp,
        "bcc": _bcc,
        "ico": _icosahedral,
    }
    scales = [0.85, 1.0, 1.15, 1.3]

    # Generate initial structures
    candidates = []
    for name, gen in prototypes.items():
        for s in scales:
            try:
                pos = gen(n) * s
            except Exception:
                continue
            if _valid(pos):
                candidates.append(pos)

    # Add random and shell structures
    for _ in range(4):
        pos = _random_compact(n, rng)
        if _valid(pos):
            candidates.append(pos)
    for _ in range(2):
        pos = _sphere_shell(n, rng)
        if _valid(pos):
            candidates.append(pos)

    if not candidates:
        pos = _fcc(n)
        e, best_pos = relax(pos, potential, kind)
        return best_pos

    # Phase 1: Parallel relaxation of all candidates
    best_e = np.inf
    best_pos = None
    relaxed = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for pos in candidates:
            if _remaining() < 15.0:
                break
            futures.append(executor.submit(_relax_soft_hard, pos, potential, kind))
        for fut in as_completed(futures):
            try:
                e, p = fut.result()
            except Exception:
                continue
            if _valid(p) and e < best_e:
                best_e = e
                best_pos = p
            relaxed.append((e, p))

    if best_pos is None:
        pos = _fcc(n)
        e, best_pos = relax(pos, potential, kind)
        best_e = e
        return best_pos

    # Phase 2: Basin hopping from the best few relaxed structures
    relaxed.sort(key=lambda x: x[0])
    for i, (e_init, p_init) in enumerate(relaxed[:3]):
        if _remaining() < 20.0:
            break
        if e_init > best_e * 1.01:
            continue
        e_bh, p_bh = _basin_hop(p_init, potential, kind, rng, max_iters=25, base_scale=0.22)
        if _valid(p_bh) and e_bh < best_e:
            best_e = e_bh
            best_pos = p_bh

    # Phase 3: One more round of basin hopping from the global best with smaller steps
    if _remaining() > 15.0:
        e_bh, p_bh = _basin_hop(best_pos, potential, kind, rng, max_iters=15, base_scale=0.15)
        if _valid(p_bh) and e_bh < best_e:
            best_e = e_bh
            best_pos = p_bh

    return best_pos
# EVOLVE-BLOCK-END
