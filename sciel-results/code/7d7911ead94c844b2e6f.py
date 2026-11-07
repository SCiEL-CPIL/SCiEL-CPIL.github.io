"""Sutton-Chen transition-metal cluster global minima search.
Multi-start with structurally informed initializations, strict deadline guard,
and a final gradient-based polish pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except Exception:
    relax = None
    sutton_chen_energy_gradient = None

# Hard wall-clock budget (evaluator kills at 330s; we target 300s with margin)
_HARD_LIMIT = 300.0
_SAFETY = 12.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _time_left():
    return max(0.0, _HARD_LIMIT - _SAFETY - _elapsed())


def _validate(positions):
    """Vectorized validity check: finite + min pair distance >= 0.35."""
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = positions.shape[0]
    if n < 2:
        return True
    # Pairwise distances via broadcasting (n up to ~80 -> 6400 pairs, fine)
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    return bool(np.min(d2) >= 0.35 ** 2)


def _energy_check(positions, kind):
    """Return energy if valid, else None."""
    if not _validate(positions):
        return None
    try:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        if np.isfinite(e):
            return float(e)
    except Exception:
        pass
    return None


# ---------- Initialization generators ----------

def _icosahedral(n, spacing=0.72, rng=None):
    """Icosahedral shells: center + 12 + 30 + 92..."""
    if rng is None:
        rng = np.random.default_rng(0)
    phi = (1 + np.sqrt(5)) / 2
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    positions = [np.zeros(3)]
    # Shell 1: 12 vertices at radius r1
    r1 = spacing
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(r1 * v)

    # Shell 2: 30 vertices (midpoints of icosahedron edges, pushed out)
    if len(positions) < n:
        # Edge midpoints of icosahedron
        icos_edges = [
            (0, 1), (0, 4), (0, 5), (0, 8), (0, 9),
            (1, 2), (1, 4), (1, 5), (1, 10), (1, 11),
            (2, 3), (2, 4), (2, 5), (2, 8), (2, 9),
            (3, 4), (3, 5), (3, 10), (3, 11),
            (4, 7), (4, 8), (4, 9),
            (5, 6), (5, 8), (5, 9),
            (6, 7), (6, 10), (6, 11),
            (7, 10), (7, 11),
        ]
        shell2 = []
        for a, b in icos_edges:
            m = (verts[a] + verts[b]) / 2.0
            m = m / np.linalg.norm(m)
            shell2.append(m)
        shell2 = np.array(shell2)
        # Deduplicate
        unique = []
        for v in shell2:
            if not any(np.allclose(v, u, atol=1e-8) for u in unique):
                unique.append(v)
        shell2 = np.array(unique)
        r2 = spacing * 1.7
        for v in shell2:
            if len(positions) >= n:
                break
            positions.append(r2 * v)

    # Fill with random if needed
    while len(positions) < n:
        p = rng.uniform(-spacing * 3, spacing * 3, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
        else:
            positions.append(p)  # allow slight overlap; validation will catch
    return np.array(positions[:n])


def _fcc_cluster(n, spacing=0.72, rng=None):
    """FCC-like cluster: build from FCC lattice, select closest n to origin."""
    a = spacing
    max_c = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 3
    pts = []
    for ix in range(-max_c, max_c + 1):
        for iy in range(-max_c, max_c + 1):
            for iz in range(-max_c, max_c + 1):
                pts.append(np.array([ix * a, iy * a, iz * a]))
                pts.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                pts.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                pts.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    # Greedy selection with min distance
    selected = []
    min_d = 0.5 * a
    for idx in order:
        if len(selected) >= n:
            break
        p = pts[idx]
        ok = True
        for s in selected:
            if np.linalg.norm(p - s) < min_d:
                ok = False
                break
        if ok:
            selected.append(p)
    while len(selected) < n:
        selected.append(pts[order[len(selected) % len(order)]])
    return np.array(selected[:n])


def _bcc_cluster(n, spacing=0.72, rng=None):
    """BCC-like cluster."""
    a = spacing
    max_c = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 3
    pts = []
    for ix in range(-max_c, max_c + 1):
        for iy in range(-max_c, max_c + 1):
            for iz in range(-max_c, max_c + 1):
                pts.append(np.array([ix * a, iy * a, iz * a]))
                pts.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    selected = []
    min_d = 0.5 * a
    for idx in order:
        if len(selected) >= n:
            break
        p = pts[idx]
        ok = True
        for s in selected:
            if np.linalg.norm(p - s) < min_d:
                ok = False
                break
        if ok:
            selected.append(p)
    while len(selected) < n:
        selected.append(pts[order[len(selected) % len(order)]])
    return np.array(selected[:n])


def _random_compact(n, spacing=0.72, rng=None):
    """Random compact cluster via rejection sampling."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 300
    min_d = 0.4 * spacing
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    # Fill remaining
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _deformed(base, rng, scale=0.08):
    """Add small random perturbation to a base configuration."""
    return base + rng.normal(0, scale, base.shape)


def _scaled(base, factor):
    """Scale a configuration about its centroid."""
    centroid = base.mean(axis=0)
    return (base - centroid) * factor + centroid


def _generate_starts(n, kind, rng):
    """Generate diverse starting configurations."""
    spacing = 0.72
    starts = []
    seen = []

    def _add(pos):
        if len(pos) != n or not np.all(np.isfinite(pos)):
            return
        # Deduplicate: check min distance to existing starts
        for s in seen:
            d = np.linalg.norm(pos - s, axis=1)
            if np.min(d) < 0.1:
                return
        seen.append(pos)
        starts.append(pos)

    # Icosahedral
    try:
        _add(_icosahedral(n, spacing, rng))
    except Exception:
        pass

    # FCC
    try:
        _add(_fcc_cluster(n, spacing, rng))
    except Exception:
        pass

    # BCC
    try:
        _add(_bcc_cluster(n, spacing, rng))
    except Exception:
        pass

    # Random compact (4 variants)
    for _ in range(4):
        try:
            _add(_random_compact(n, spacing, rng))
        except Exception:
            pass

    # Deformed icosahedral
    try:
        icosa = _icosahedral(n, spacing, rng)
        _add(_deformed(icosa, rng, 0.06))
        _add(_deformed(icosa, rng, 0.12))
    except Exception:
        pass

    # Scaled FCC
    try:
        fcc = _fcc_cluster(n, spacing, rng)
        _add(_scaled(fcc, 0.95))
        _add(_scaled(fcc, 1.05))
    except Exception:
        pass

    # More random
    for _ in range(3):
        try:
            _add(_random_compact(n, spacing, rng))
        except Exception:
            pass

    return starts


def _relax_start(start, kind, time_budget):
    """Relax a single start with a time budget. Returns (energy, positions) or (inf, None)."""
    if not _validate(start):
        return np.inf, None
    if _time_left() < 2.0:
        return np.inf, None
    try:
        energy, positions = relax(start, 'sutton_chen', kind)
        if positions is not None and np.all(np.isfinite(positions)) and _validate(positions):
            if np.isfinite(energy):
                return float(energy), positions
    except Exception:
        pass
    return np.inf, None


def _polish(positions, kind):
    """Final polish: one more L-BFGS-B call with tighter settings if time permits."""
    if _time_left() < 3.0:
        return positions
    try:
        energy, positions2 = relax(positions, 'sutton_chen', kind)
        if positions2 is not None and np.all(np.isfinite(positions2)) and _validate(positions2):
            return positions2
    except Exception:
        pass
    return positions


def _fallback_valid(n, rng):
    """Generate a guaranteed-valid configuration (grid with jitter)."""
    # Place atoms on a 3D grid with spacing 0.5 (well above 0.35)
    spacing = 0.5
    side = int(np.ceil(n ** (1.0 / 3.0))) + 1
    pts = []
    for ix in range(-side, side + 1):
        for iy in range(-side, side + 1):
            for iz in range(-side, side + 1):
                if len(pts) >= n:
                    break
                pts.append(np.array([ix * spacing, iy * spacing, iz * spacing]))
        if len(pts) >= n:
            break
        if len(pts) >= n:
            break
    while len(pts) < n:
        pts.append(rng.uniform(-3, 3, 3))
    pts = np.array(pts[:n])
    # Enforce min distance
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 0.35:
                direction = pts[i] - pts[j]
                if np.linalg.norm(direction) > 1e-10:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                pts[j] = pts[i] + direction * 0.36
    return pts


def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _T0
    _T0 = time.time()  # Reset clock per call

    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)

    # Generate diverse starts
    starts = _generate_starts(n, kind, rng)
    if not starts:
        starts = [_random_compact(n, 0.72, rng)]

    best_energy = np.inf
    best_positions = starts[0]

    # Relax each start
    for i, start in enumerate(starts):
        if _time_left() < 8.0:
            break
        e, pos = _relax_start(start, kind, None)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos

    # Use remaining time for additional random starts
    max_extra = 15
    attempts = 0
    while _time_left() > 12.0 and attempts < max_extra:
        attempts += 1
        start = _random_compact(n, 0.72, rng)
        e, pos = _relax_start(start, kind, None)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos

    # Final polish
    if _time_left() > 5.0 and best_positions is not None:
        best_positions = _polish(best_positions, kind)

    # Final validation and fallback
    if not _validate(best_positions):
        best_positions = _fallback_valid(n, rng)

    return best_positions
# EVOLVE-BLOCK-END
