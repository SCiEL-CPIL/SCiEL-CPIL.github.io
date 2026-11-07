# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard: evaluator kills at ~330s wall clock. Keep a large margin.
_DEADLINE = time.time() + 285.0
_MIN_DIST = 0.35


def _time_left():
    return _DEADLINE - time.time()


def _valid(pos):
    """Finite coords and every pair at least _MIN_DIST apart."""
    if pos is None:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    d = pos[:, None, :] - pos[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(d2, np.inf)
    return np.min(d2) >= _MIN_DIST ** 2


def _bcc_positions(n):
    """Fill BCC lattice shells outwards, taking the first n sites."""
    sites = [(0, 0, 0)]
    # Shell k: cube corners (2k)^3 plus face-centers (2k+1)^2 per face.
    k = 1
    while len(sites) < n:
        for x in (-k, k):
            for y in (-k, k):
                for z in (-k, k):
                    sites.append((x, y, z))
        for face in range(3):
            for a in range(-k, k + 1):
                for b in range(-k, k + 1):
                    if face == 0:
                        pts = [(a, b, k), (a, b, -k)]
                    elif face == 1:
                        pts = [(a, k, b), (a, -k, b)]
                    else:
                        pts = [(k, a, b), (-k, a, b)]
                    sites.extend(pts)
        k += 1
    sites = sites[:n]
    # Center of mass to origin
    arr = np.array(sites, dtype=float)
    arr -= arr.mean(axis=0)
    return arr


def _fcc_positions(n):
    """Fill FCC lattice (concentric shells) outwards, taking the first n sites."""
    sites = [(0, 0, 0)]
    k = 1
    while len(sites) < n:
        # Cube corners (2k)^3
        for x in (-k, k):
            for y in (-k, k):
                for z in (-k, k):
                    sites.append((x, y, z))
        # Face centers (2k+1)^2 per face
        for face in range(3):
            for a in range(-k, k + 1):
                for b in range(-k, k + 1):
                    if face == 0:
                        pts = [(a, b, k), (a, b, -k)]
                    elif face == 1:
                        pts = [(a, k, b), (a, -k, b)]
                    else:
                        pts = [(k, a, b), (-k, a, b)]
                    sites.extend(pts)
        k += 1
    sites = sites[:n]
    arr = np.array(sites, dtype=float)
    arr -= arr.mean(axis=0)
    return arr


def _random_compact(n, rng, spacing=0.9):
    """Random compact cluster with a minimum separation to avoid overlaps."""
    radius = spacing * n ** (1.0 / 3.0) * 0.95
    pts = []
    attempts = 0
    min_sep = 0.7 * spacing
    while len(pts) < n:
        attempts += 1
        if attempts > 50000:
            # Fallback: place on a coarse grid to guarantee termination
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    if len(pts) < n:
        # Grid fallback
        g = []
        step = spacing * 0.9
        L = int(np.ceil(n ** (1.0 / 3.0))) + 2
        for ix in range(-L, L + 1):
            for iy in range(-L, L + 1):
                for iz in range(-L, L + 1):
                    g.append([ix * step, iy * step, iz * step])
                    if len(g) >= n:
                        break
                if len(g) >= n:
                    break
            if len(g) >= n:
                break
        pts = g[:n]
    arr = np.array(pts, dtype=float)
    arr -= arr.mean(axis=0)
    return arr


def _relax_one(pos, potential, kind):
    try:
        e, p = relax(pos, potential, kind)
    except Exception:
        return None, None
    if not np.isfinite(e):
        return None, None
    if not _valid(p):
        return None, None
    return float(e), p


def construct(instance):
    n = int(instance["n"])
    potential = instance.get("potential", "sutton_chen")
    kind = instance.get("kind")
    if kind is None:
        kind = instance.get("rho")

    best_e = np.inf
    best_p = None

    def consider(e, p):
        nonlocal best_e, best_p
        if e is not None and _valid(p) and e < best_e:
            best_e = e
            best_p = p

    # 1) Structured lattice starts (BCC and FCC) — strongest basins.
    for gen in (_bcc_positions, _fcc_positions):
        if _time_left() < 20.0:
            break
        try:
            start = gen(n)
        except Exception:
            continue
        if _valid(start):
            e, p = _relax_one(start, potential, kind)
            consider(e, p)

    # 2) Random compact starts, scaled by remaining time.
    rng = np.random.default_rng(n * 1000 + 7)
    n_random = 8
    while _time_left() > 30.0 and n_random > 0:
        start = _random_compact(n, rng)
        e, p = _relax_one(start, potential, kind)
        consider(e, p)
        n_random -= 1

    # 3) If we still have generous time, add a few perturbed random starts.
    if _time_left() > 90.0:
        for _ in range(4):
            if _time_left() < 30.0:
                break
            start = _random_compact(n, rng)
            start = start + 0.05 * rng.standard_normal(start.shape)
            e, p = _relax_one(start, potential, kind)
            consider(e, p)

    # Fallback: ensure we always return something valid.
    if best_p is None:
        start = _bcc_positions(n)
        e, p = _relax_one(start, potential, kind)
        if e is not None:
            best_e, best_p = e, p
        else:
            # Last-resort: a valid grid cluster (may be high energy but valid).
            spacing = 0.9
            L = int(np.ceil(n ** (1.0 / 3.0))) + 2
            pts = []
            for ix in range(-L, L + 1):
                for iy in range(-L, L + 1):
                    for iz in range(-L, L + 1):
                        pts.append([ix * spacing, iy * spacing, iz * spacing])
                        if len(pts) >= n:
                            break
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            best_p = np.array(pts[:n], dtype=float)
            best_p -= best_p.mean(axis=0)

    return best_p
# EVOLVE-BLOCK-END
