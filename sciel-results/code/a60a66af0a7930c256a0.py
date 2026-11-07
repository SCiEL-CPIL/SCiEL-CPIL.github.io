"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    positions = np.asarray(positions)
    if positions.size == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = positions.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if d[0] < min_dist:
            return False
    return True


def _scale_cluster(pts, n):
    pts = np.asarray(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    if n > 1:
        rmax = np.max(dists[order[:n]])
        if rmax > 0:
            pts = pts * (0.72 * n ** (1.0 / 3.0) * 0.6) / rmax
    return pts


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append([i + dx, j + dy, k + dz])
    pts = np.array(pts)
    return _scale_cluster(pts, n)


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts)
    return _scale_cluster(pts, n)


def _icosahedron_data():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                edges.append((i, j))
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                v1, v2, v3 = verts[i], verts[j], verts[k]
                d1 = np.linalg.norm(v2 - v1)
                d2 = np.linalg.norm(v3 - v1)
                d3 = np.linalg.norm(v3 - v2)
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    faces.append((i, j, k))
    return verts, edges, faces


def _make_mackay(n):
    """Mackay icosahedral shells: 1, 13, 55, 147, 309, 561 atoms."""
    verts, edges, faces = _icosahedron_data()
    pts = [np.zeros(3)]
    # Shell 1 (12): vertices at r=1
    for v in verts:
        pts.append(v)
    # Shell 2 (42): 30 edge midpoints + 12 face-centre directions, r=sqrt(2+phi)
    phi = (1 + np.sqrt(5)) / 2
    r2 = np.sqrt(2 + phi)
    for (i, j) in edges:
        e = (verts[i] + verts[j]) / 2
        e = e / np.linalg.norm(e) * r2
        pts.append(e)
    for (i, j, k) in faces:
        c = (verts[i] + verts[j] + verts[k]) / 3
        c = c / np.linalg.norm(c) * r2
        pts.append(c)
    # Shell 3 (92): vertices + edge midpoints + face centres at r=sqrt(5+2phi)
    r3 = np.sqrt(5 + 2 * phi)
    for v in verts:
        pts.append(v * r3)
    for (i, j) in edges:
        e = (verts[i] + verts[j]) / 2
        e = e / np.linalg.norm(e) * r3
        pts.append(e)
    for (i, j, k) in faces:
        c = (verts[i] + verts[j] + verts[k]) / 3
        c = c / np.linalg.norm(c) * r3
        pts.append(c)
    # Shell 4 (252) via uniform sphere points, r=sqrt(14+6phi)
    r4 = np.sqrt(14 + 6 * phi)
    golden = (1 + np.sqrt(5)) / 2
    N4 = 252
    for m in range(N4):
        y = 1 - 2 * (m + 0.5) / N4
        rad = np.sqrt(max(0.0, 1 - y * y))
        th = 2 * np.pi * m / golden
        p = np.array([np.cos(th) * rad, y, np.sin(th) * rad])
        pts.append(p * r4)
    pts = np.array(pts)
    # de-duplicate
    unique = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 0.05 for q in unique):
            unique.append(p)
    pts = np.array(unique)
    return _scale_cluster(pts, n)


def _make_random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 8000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale):
    return positions + rng.normal(0, scale, positions.shape)


def _grow_to_target(start, kind, target, rng):
    """Sequentially add atoms to low-energy interstitial sites of the relaxed
    (k-1)-cluster; relax after each addition."""
    current = np.asarray(start, dtype=float).copy()
    cur_n = current.shape[0]
    for k in range(cur_n + 1, target + 1):
        if _time_left() < 25.0:
            break
        if not _is_valid(current):
            break
        # candidate insertion sites: near existing atoms (interstitial) + a few bulk points
        cands = []
        for i in range(current.shape[0]):
            d = np.linalg.norm(current[i])
            if d < 1e-8:
                continue
            dirv = current[i] / d
            for off in (0.30, 0.45, 0.60):
                cands.append(current[i] + dirv * off)
        for i in range(current.shape[0]):
            for j in range(i + 1, current.shape[0]):
                mid = (current[i] + current[j]) / 2
                cands.append(mid)
        cands = np.array(cands)
        # keep only candidates that are reasonably separated from the cluster
        keep = []
        for c in cands:
            if np.linalg.norm(c) > 0.3:
                keep.append(c)
        keep = np.array(keep) if len(keep) else cands
        # score each candidate by the relaxed energy of the (k)-cluster
        best_e = np.inf
        best_pos = None
        for c in keep:
            trial = np.vstack([current, c[None, :]])
            if not _is_valid(trial):
                continue
            try:
                e, pos = relax(trial, 'sutton_chen', kind)
            except Exception:
                continue
            if np.isfinite(e) and _is_valid(pos) and e < best_e:
                best_e = e
                best_pos = pos
        if best_pos is not None:
            current = best_pos
        else:
            # fallback: add a random point and relax
            rmax = np.max(np.linalg.norm(current, axis=1)) if current.shape[0] else 1.0
            c = rng.normal(0, max(0.5, 0.3 * rmax), 3)
            trial = np.vstack([current, c[None, :]])
            try:
                e, pos = relax(trial, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(pos):
                    current = pos
                else:
                    break
            except Exception:
                break
    return current


def _basin_hop(positions, kind, rng, n_hops, base_scale):
    """Simulated-annealing basin hopping over L-BFGS-B basins."""
    cur = np.asarray(positions, dtype=float).copy()
    try:
        cur_e, _ = sutton_chen_energy_gradient(cur, kind)
    except Exception:
        cur_e = np.inf
    if not np.isfinite(cur_e):
        try:
            cur_e, cur = relax(cur, 'sutton_chen', kind)
        except Exception:
            return cur
    best_e, best_pos = cur_e, cur.copy()
    T0 = 0.05
    for hop in range(n_hops):
        if _time_left() < 12.0:
            break
        scale = base_scale * (0.9 ** hop)
        pert = cur + rng.normal(0, scale, cur.shape)
        if not _is_valid(pert):
            continue
        try:
            e_new, pos_new = relax(pert, 'sutton_chen', kind)
        except Exception:
            continue
        if not (np.isfinite(e_new) and _is_valid(pos_new)):
            continue
        T = T0 * (0.85 ** hop)
        if e_new <= cur_e or (rng.random() < np.exp(-(e_new - cur_e) / max(T, 1e-9))):
            cur, cur_e = pos_new, e_new
        if e_new < best_e:
            best_e, best_pos = e_new, pos_new.copy()
    return best_pos


def _relax_valid(pos, kind):
    try:
        e, p = relax(pos, 'sutton_chen', kind)
        if np.isfinite(e) and _is_valid(p):
            return e, p
    except Exception:
        pass
    return None, None


def construct(instance):
    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345)

    best_energy = np.inf
    best_positions = None

    def consider(pos, e=None):
        nonlocal best_energy, best_positions
        if pos is None:
            return
        if not _is_valid(pos):
            return
        if e is None or not np.isfinite(e):
            e, pos = _relax_valid(pos, kind)
            if pos is None:
                return
        if np.isfinite(e) and e < best_energy:
            best_energy = e
            best_positions = pos.copy()

    # 1) deterministic structural starts (FCC, BCC, Mackay)
    for gen in (_make_fcc, _make_bcc, _make_mackay):
        if _time_left() < 30.0:
            break
        try:
            s = gen(n)
        except Exception:
            continue
        e, p = _relax_valid(s, kind)
        consider(p, e)

    # 2) deterministic growth: build up to target from a small seed
    if _time_left() > 90.0:
        seed_n = min(12, n)
        try:
            seed = _make_fcc(seed_n)
        except Exception:
            seed = rng.uniform(-0.5, 0.5, (seed_n, 3))
        try:
            grown = _grow_to_target(seed, kind, n, rng)
            e, p = _relax_valid(grown, kind)
            consider(p, e)
        except Exception:
            pass

    # 3) random compact starts
    for _ in range(2):
        if _time_left() < 30.0:
            break
        rc = _make_random_compact(n, rng)
        if rc is not None:
            e, p = _relax_valid(rc, kind)
            consider(p, e)

    # 4) basin-hopping refinement from the best (and a couple of good starts)
    if _time_left() > 40.0:
        bh_starts = []
        for gen in (_make_fcc, _make_mackay):
            try:
                s = gen(n)
                if _is_valid(s):
                    bh_starts.append(s)
            except Exception:
                pass
        if best_positions is not None:
            bh_starts.append(best_positions)
        for s in bh_starts:
            if _time_left() < 25.0:
                break
            try:
                refined = _basin_hop(s, kind, rng, n_hops=8, base_scale=0.18)
                e, p = _relax_valid(refined, kind)
                consider(p, e)
            except Exception:
                pass

    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
