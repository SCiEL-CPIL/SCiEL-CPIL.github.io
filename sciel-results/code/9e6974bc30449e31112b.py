# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov parameters
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16.0


def _dzugutov_energy(pos):
    """Vectorised Dzugutov energy; returns +inf if any pair is < 0.5 apart."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    d = np.triu(d, k=1)
    if d.min() < 0.5:
        return np.inf
    e = np.zeros(d.shape)
    m1 = d < _a
    m2 = (d >= _a) & (d < _b)
    r1 = d[m1]
    e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    r2 = d[m2]
    e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(e.sum())


def _rotmat(axis, ang):
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:
        return np.eye(3)
    axis = axis / nrm
    x, y, z = axis
    c = np.cos(ang)
    s = np.sin(ang)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _icosahedron(scale=1.0):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v = v / np.linalg.norm(v[0]) * scale
    return np.vstack([np.zeros(3), v])  # 13 atoms


def _decahedron(scale=1.0):
    """12-vertex truncated icosahedron + centre = 13 atoms (5-fold, 3-fold faces)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v = v / np.linalg.norm(v[0]) * scale
    return np.vstack([np.zeros(3), v])


def _mckay55(scale=1.0):
    """Mackay 55: 13-atom icosahedra at the 55-shell sites (1 centre + 12 + 30 + 12)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices (icosahedron)
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    # 30 edge midpoints
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (8, 9), (9, 11), (11, 10), (10, 8),
             (0, 4), (0, 5), (0, 8), (0, 9), (1, 4), (1, 5), (1, 8), (1, 9),
             (2, 6), (2, 7), (2, 10), (2, 11), (3, 6), (3, 7), (3, 10), (3, 11),
             (4, 8), (4, 9), (5, 8), (5, 9), (6, 10), (6, 11), (7, 10), (7, 11)]
    mids = np.array([(verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
                     for i, j in edges])
    sites = np.vstack([np.zeros(3), verts, mids, verts])  # 1 + 12 + 30 + 12 = 55
    sites = sites / np.linalg.norm(verts[0]) * scale
    # place 13-atom icosahedron (unit, centre at origin) at each site
    unit = _icosahedron(1.0)
    out = []
    for s in sites:
        out.append(unit + s)
    return np.vstack(out)


def _mckay55_shell(scale=1.0):
    """The 42 outer atoms of the 55-shell (12 + 30) as icosahedron centres, i.e. a
    42-atom 'fragment' that can be merged with a 13-atom core to form a 55 Mackay."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (8, 9), (9, 11), (11, 10), (10, 8),
             (0, 4), (0, 5), (0, 8), (0, 9), (1, 4), (1, 5), (1, 8), (1, 9),
             (2, 6), (2, 7), (2, 10), (2, 11), (3, 6), (3, 7), (3, 10), (3, 11),
             (4, 8), (4, 9), (5, 8), (5, 9), (6, 10), (6, 11), (7, 10), (7, 11)]
    mids = np.array([(verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
                     for i, j in edges])
    sites = np.vstack([verts, mids])  # 42
    sites = sites / np.linalg.norm(verts[0]) * scale
    return sites


def _build_cluster(n, rng):
    """Pick a set of polytetrahedral fragments whose atom counts sum to n,
    place them on a coarse shell, and return (positions, [fragment atom-counts])."""
    # candidate fragment sizes (all 13, 13, 55, 42+13=55, etc.)
    frags = [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
    # greedy: use 55 blocks where possible, fill with 13
    blocks = []
    rem = n
    # prefer 55-mackay if it divides nicely
    while rem >= 55 and len(blocks) < 4:
        blocks.append(55)
        rem -= 55
    while rem >= 13:
        blocks.append(13)
        rem -= 13
    # fix remainder by adjusting a 13 -> smaller not available; instead pad by
    # splitting the last block into a 13 + (rem) and just place rem extra atoms
    # as a small icosahedron fragment of size rem (if rem<13 use rem directly)
    if rem > 0:
        blocks.append(rem)
    # recompute to ensure sum == n
    total = sum(blocks)
    if total != n:
        # adjust: replace last block to make sum exact
        blocks[-1] = rem
        total = sum(blocks)
        if total != n:
            # fallback: build from 13s + remainder
            blocks = [13] * (n // 13)
            r = n % 13
            if r:
                blocks.append(r)
    # place block centres on a coarse spherical shell
    k = len(blocks)
    if k == 1:
        centres = [np.zeros(3)]
    else:
        # fibonacci sphere for centres, radius grows with total size
        radius = 1.15 * n ** (1.0 / 3.0) * 0.9
        idx = np.arange(k)
        phi = np.pi * (3.0 - np.sqrt(5.0))
        y = 1.0 - (idx / max(k - 1, 1)) * 2.0
        r = np.sqrt(np.clip(1.0 - y * y, 0, 1))
        th = phi * idx
        centres = np.column_stack([r * np.cos(th), y, r * np.sin(th)]) * radius
    # assemble
    pos_list = []
    counts = []
    for ci, cnt in zip(centres, blocks):
        if cnt == 55:
            frag = _mckay55(1.0)
        elif cnt == 13:
            frag = _icosahedron(1.0)
        elif cnt == 42:
            frag = _mckay55_shell(1.0)
        else:
            # small fragment: icosahedron truncated to cnt, or a ring
            if cnt <= 13:
                frag = _icosahedron(1.0)[:cnt]
            else:
                frag = _mckay55(1.0)[:cnt]
        pos_list.append(frag + ci)
        counts.append(len(frag))
    pos = np.vstack(pos_list)
    return pos, counts


def _relax_and_check(pos, potential, parameter):
    try:
        e, p = relax(pos, potential, parameter)
    except Exception:
        return None, None
    if p is None or len(p) != len(pos):
        return None, None
    if not np.all(np.isfinite(p)):
        return None, None
    d = cdist(p, p)
    np.fill_diagonal(d, np.inf)
    if np.triu(d, 1).min() < 0.5 - 1e-6:
        return None, None
    return e, p


def _anneal_blocks(pos, blocks, potential, parameter, rng, budget):
    """Simulated annealing over block translations (+ small rotations) with
    full local relaxation after each accepted move."""
    # split pos into blocks by the recorded sizes
    offsets = np.cumsum([0] + blocks)
    segs = [pos[offsets[i]:offsets[i + 1]] for i in range(len(blocks))]
    best = _relax_and_check(pos, potential, parameter)
    if best is None or best[0] is None:
        return pos
    best_e, best_p = best
    T0 = 2.0
    Tend = 0.02
    n_iter = max(6, int(budget // 1.5))
    for it in range(n_iter):
        frac = it / max(n_iter - 1, 1)
        T = T0 * (Tend / T0) ** frac
        # pick a random block to move
        bi = rng.integers(len(segs))
        # random small translation
        dv = rng.normal(0, 0.6, 3)
        # random small rotation about random axis
        ang = rng.normal(0, 0.3)
        axis = rng.normal(0, 1, 3)
        R = _rotmat(axis, ang)
        new_segs = [s.copy() for s in segs]
        new_segs[bi] = (new_segs[bi] - new_segs[bi].mean(0)) @ R.T + new_segs[bi].mean(0) + dv
        new_pos = np.vstack(new_segs)
        e_new, p_new = _relax_and_check(new_pos, potential, parameter)
        if e_new is None or not np.isfinite(e_new):
            continue
        if e_new < best_e or rng.random() < np.exp(-(e_new - best_e) / max(T, 1e-6)):
            # accept
            segs = [s.copy() for s in new_segs]
            pos = p_new.copy()
            # re-split by sizes (p_new is the relaxed full cluster, order preserved)
            segs = [pos[offsets[i]:offsets[i + 1]] for i in range(len(blocks))]
            best_e, best_p = e_new, p_new
    return best_p


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    t0 = time.time()
    deadline = t0 + 45.0  # per-instance budget (6 * 45 = 270 < 330)
    rng = np.random.default_rng(12345 + size)

    best_e = np.inf
    best_p = None

    def consider(pos):
        nonlocal best_e, best_p
        e, p = _relax_and_check(pos, potential, parameter)
        if e is None or not np.isfinite(e):
            return
        if e < best_e:
            best_e, best_p = e, p

    # --- Phase 1: direct polytetrahedral templates ---
    if size == 55:
        consider(_mckay55(1.0))
    if size == 13:
        consider(_icosahedron(1.0))

    # --- Phase 2: fragment-merger with coarse annealing ---
    n_templates = 6
    for ti in range(n_templates):
        if time.time() > deadline - 10.0:
            break
        pos, counts = _build_cluster(size, rng)
        # light initial relaxation
        e, p = _relax_and_check(pos, potential, parameter)
        if e is None or not np.isfinite(e):
            continue
        # anneal the block arrangement
        remaining = deadline - time.time() - 5.0
        if remaining > 3.0:
            p2 = _anneal_blocks(p, counts, potential, parameter, rng, remaining)
            e2, p2c = _relax_and_check(p2, potential, parameter)
            if e2 is not None and np.isfinite(e2):
                consider(p2c)
        else:
            consider(p)

    # --- Phase 3: safety fallback (random compact) if nothing valid ---
    if best_p is None:
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        pts = []
        while len(pts) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.9 for q in pts):
                pts.append(p)
        consider(np.array(pts))

    # final guard
    if best_p is None:
        # absolute fallback: linear chain (valid, high energy)
        best_p = np.array([[i * 0.7, 0.0, 0.0] for i in range(size)], dtype=float)
    return best_p
# EVOLVE-BLOCK-END
