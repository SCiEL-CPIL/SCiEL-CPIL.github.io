"""Sutton-Chen transition-metal cluster global minima: multi-start from diverse
structured + jittered compact seeds, each relaxed with L-BFGS-B, under a strict
wall-clock deadline. Returns the (n,3) coordinates of the best relaxed cluster."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 315.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators: produce compact, valid (min-pair-distance) (n,3) arrays.
# All are centered near the origin and scaled so typical nearest-neighbor
# distances are ~ the equilibrium spacing of the potential.
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    """Fill the fcc lattice with the n sites closest to the origin."""
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    cand = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cell = np.array([i, j, k])
                for b in basis:
                    p = (cell + b)
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    # center
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    """Fill an hcp-like lattice (two interpenetrating triangular layers)."""
    # triangular lattice in plane, stacking AB
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a  # layer spacing for hcp
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    cand = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * h
        # A layers at (i,j) integer triangular; B layers shifted
        is_b = (layer % 2 != 0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3.0) / 2.0) * j
                if is_b:
                    x += a * 0.5
                    y += a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _icosahedral_seed(n, rng, scale=1.0):
    """Start from an icosahedron and add shells / fill interior by nearest
    available point on a fine icosahedral-fcc hybrid. Fallback: fcc."""
    # Build a set of points: icosahedron vertices + a few inner shells of an
    # fcc lattice, then take the n closest to origin.
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts_list = [p * 1.0 for p in ico]
    # add fcc points to fill up to n
    fcc = _fcc_seed(n, rng, scale=1.0)
    # combine: ico first, then fcc points not too close to existing
    all_pts = list(pts_list)
    for p in fcc:
        if len(all_pts) >= n:
            break
        ok = True
        for q in all_pts:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            all_pts.append(p)
    if len(all_pts) < n:
        # top up with fcc
        for p in fcc:
            if len(all_pts) >= n:
                break
            all_pts.append(p)
    pts = np.array(all_pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _decahedral_seed(n, rng, scale=1.0):
    """Generate a decahedral-like cluster (twinned icosahedron)."""
    # Start with icosahedral, then add a twin by reflecting across a plane
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    pts_list = [p * 1.0 for p in ico]
    # Add twin: reflect across x=0 plane and offset slightly
    for p in ico:
        p_twin = np.array([-p[0], p[1], p[2]]) * 1.2
        if len(pts_list) < n:
            pts_list.append(p_twin)
    
    # Fill remaining with fcc
    fcc = _fcc_seed(n, rng, scale=1.0)
    for p in fcc:
        if len(pts_list) >= n:
            break
        ok = True
        for q in pts_list:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            pts_list.append(p)
    
    if len(pts_list) < n:
        for p in fcc:
            if len(pts_list) >= n:
                break
            pts_list.append(p)
    
    pts = np.array(pts_list[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    """Random points in a ball with a minimum-pair-distance constraint."""
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 20000
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    # if we couldn't place all, relax the constraint progressively
    if len(pts) < n:
        md = min_dist
        while len(pts) < n:
            md *= 0.95
            if md < 0.36:
                # last resort: random in ball without constraint
                while len(pts) < n:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                break
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= md for q in pts):
                    pts.append(p)
    pts = np.array(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    return pts


def _jitter(seed, rng, amount):
    return seed + rng.normal(0.0, amount, seed.shape)


def _is_valid(pts, min_dist=0.35):
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    if n < 2:
        return True
    # pairwise min distance via broadcasting (n up to ~80, fine)
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist


def _try_relax(seed, potential, kind, maxiter=2000):
    """Relax a seed; return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    parameter = kind

    # equilibrium nearest-neighbor spacing ~ 1.0 for these potentials;
    # use a scale that keeps seeds compact.
    base_scale = 1.0

    rng = np.random.default_rng(12345 + n)

    # Build a diverse list of seeds, ordered to spend budget on the most
    # promising first (structured), then jittered structured, then random.
    seeds = []

    # 1. Pure structured seeds (small jitter to escape exact lattice degeneracy)
    for gen in (_fcc_seed, _hcp_seed, _icosahedral_seed, _decahedral_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        # jittered variants
        for amt in (0.05, 0.12, 0.25):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    # 2. Random compact seeds
    for _ in range(6):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.6)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None

    # First, if we have any valid seed, make sure we always have a fallback.
    # Relax seeds one at a time
    for seed in seeds:
        if _remaining() < 10.0:
            break
        energy, positions = _try_relax(seed, potential, kind)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions

    # If no valid result, return a simple valid configuration
    if best_positions is None:
        # Fallback: linear arrangement
        positions = np.zeros((n, 3))
        positions[:, 0] = np.arange(n) * base_scale
        best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
