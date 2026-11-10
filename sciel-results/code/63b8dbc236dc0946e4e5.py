# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before the 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    return 12, 6, 144.41

def _get_spacing(kind):
    n, m, c = _get_params(kind)
    return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))

# ---------- Geometric seeds ----------

def _fcc_positions(n, a=1.0):
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3)/2, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1/6) * a * np.sqrt(3), z + c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _bcc_positions(n, a=1.0):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _icosahedral_positions(n, a=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a / np.sqrt(1 + phi**2)
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v)
    if n > 13:
        faces = [
            [0, 1, 8], [0, 2, 9], [1, 3, 8], [2, 4, 9],
            [1, 0, 5], [3, 2, 6], [0, 4, 5], [2, 6, 7],
            [5, 8, 10], [6, 9, 11], [8, 1, 10], [9, 3, 11],
            [10, 5, 7], [11, 6, 4], [8, 10, 0], [9, 11, 2],
            [5, 7, 1], [6, 4, 3], [10, 8, 0], [11, 9, 2]
        ]
        for f in faces:
            if len(positions) >= n:
                break
            positions.append((verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0)
        if n > 33:
            edges = []
            for i in range(12):
                for j in range(i+1, 12):
                    if np.linalg.norm(verts[i] - verts[j]) < 2.1 * a / np.sqrt(1+phi**2):
                        edges.append((i, j))
            for e in edges:
                if len(positions) >= n:
                    break
                positions.append((verts[e[0]] + verts[e[1]]) / 2.0)
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        positions.append(verts[idx] + np.random.normal(0, 0.1 * a, 3))
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _molten_cluster(n, spacing, seed, temp_scale=1.0):
    """Softer, more disordered seed: lower minimum separation to allow diverse basins."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    min_d = spacing * 0.5 * temp_scale
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in points):
            points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _deformed_fcc_positions(n, a=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = _fcc_positions(n, a)
    return base + rng.normal(0, a * 0.15, base.shape)

def _coarse_to_fine_grid(n, spacing, seed=0):
    """Dense low-energy core via progressively refined lattice filling."""
    rng = np.random.default_rng(seed)
    # start with a coarse lattice, then refine by adding points at midpoints
    side = max(2, int(np.cbrt(n))) + 2
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append(np.array([i - side/2, j - side/2, k - side/2]) * spacing * 1.2)
    # refine: insert midpoints between existing neighbors
    for _ in range(2):
        new_pts = list(pts)
        for a in range(len(pts)):
            for b in range(a+1, len(pts)):
                d = pts[a] - pts[b]
                if 0.9 * spacing < np.linalg.norm(d) < 1.6 * spacing:
                    new_pts.append((pts[a] + pts[b]) / 2.0)
        pts = new_pts
    pts.sort(key=lambda p: np.linalg.norm(p))
    # dedupe
    seen = set()
    out = []
    for p in pts:
        key = tuple(np.round(p, 5))
        if key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= n:
            break
    if len(out) < n:
        while len(out) < n:
            out.append(rng.uniform(-spacing*3, spacing*3, 3))
    return np.array(out[:n])

# ---------- Thermodynamically consistent incremental insertion ----------

def _incremental_energy(pos, new_atom, kind):
    """Approximate incremental energy of adding new_atom to the cluster pos."""
    n_exp, m_exp, c = _get_params(kind)
    if len(pos) == 0:
        return 0.0
    d = pos - new_atom
    r2 = np.sum(d * d, axis=1)
    r2 = np.maximum(r2, 1e-12)
    pair_sum = np.sum(r2 ** (-n_exp / 2.0))
    coh_sum = np.sum(r2 ** (-m_exp / 2.0))
    return 0.5 * pair_sum - c * np.sqrt(coh_sum)

def _gradient_guided_insertion(n, kind, spacing, seed=0):
    """Build cluster atom-by-atom, placing each new atom to minimize incremental
    energy. Produces a thermodynamically consistent seed (dense low-energy core)."""
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    for count in range(1, n):
        if _remaining() < 15.0:
            break
        radius = spacing * count ** (1.0/3.0) * 1.3
        best_e = np.inf
        best_p = None
        for _ in range(40):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            e = _incremental_energy(np.array(positions), p, kind)
            if e < best_e:
                best_e = e
                best_p = p
        if best_p is None:
            best_p = rng.uniform(-radius, radius, 3)
        positions.append(best_p)
    while len(positions) < n:
        positions.append(rng.uniform(-spacing*3, spacing*3, 3))
    return np.array(positions[:n])

# ---------- Validation ----------

def _min_distance_check(pos):
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    global _START_TIME, _DEADLINE
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    spacing = _get_spacing(kind)

    candidates = []
    for gen in [_fcc_positions, _hcp_positions, _bcc_positions, _icosahedral_positions]:
        try:
            c = gen(n, spacing)
            if len(c) == n and np.all(np.isfinite(c)):
                candidates.append(c)
        except Exception:
            pass
    try:
        candidates.append(_coarse_to_fine_grid(n, spacing, 7))
    except Exception:
        pass
    for seed in [42, 123, 456, 789, 101, 202]:
        try:
            candidates.append(_random_cluster(n, spacing, seed))
        except Exception:
            pass
    for seed in [11, 22]:
        try:
            candidates.append(_molten_cluster(n, spacing, seed, 1.0))
        except Exception:
            pass
    try:
        candidates.append(_deformed_fcc_positions(n, spacing, 5))
    except Exception:
        pass
    # Thermodynamically consistent insertion seeds (best new starting points)
    for seed in [0, 3]:
        if _remaining() > 25.0:
            try:
                candidates.append(_gradient_guided_insertion(n, kind, spacing, seed))
            except Exception:
                pass

    best_energy = np.inf
    best_positions = None

    tasks = [(pos, potential, kind) for pos in candidates]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_task, task): task for task in tasks}
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                energy, relaxed_pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass

    if best_positions is None:
        if _remaining() > 5.0:
            try:
                pos = _random_cluster(n, spacing, 999)
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos):
                    best_positions = relaxed_pos
            except Exception:
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]

    # Multi-scale perturbation refinement around the best structure
    if best_positions is not None and _remaining() > 20.0:
        scales = [0.4, 0.2, 0.1, 0.05, 0.03]
        current_best = best_positions.copy()
        current_energy = best_energy
        for scale in scales:
            if _remaining() < 10.0:
                break
            rng = np.random.default_rng(42)
            improved = False
            for attempt in range(5):
                if _remaining() < 5.0:
                    break
                perturbed = current_best + rng.normal(0, scale, current_best.shape)
                try:
                    energy, relaxed_pos = relax(perturbed, potential, kind)
                    if _valid_result(energy, relaxed_pos) and energy < current_energy:
                        current_energy = energy
                        current_best = relaxed_pos
                        improved = True
                except Exception:
                    pass
            if not improved and scale < 0.1:
                break

    # Final high-precision polish
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, polished_pos = relax(current_best, potential, kind)
            if _valid_result(energy, polished_pos):
                best_positions = polished_pos
                best_energy = energy
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
