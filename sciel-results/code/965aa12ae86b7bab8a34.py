"""Structure-based global search with coarse-to-fine SC relaxation for transition-metal clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

_SOFT_PARAMS = {
    '12-6': {'n': 6.0, 'm': 3.0, 'c_scale': 0.5},
    '9-6': {'n': 4.5, 'm': 3.0, 'c_scale': 0.5},
    '10-8': {'n': 5.0, 'm': 4.0, 'c_scale': 0.5},
}

# Precomputed icosahedron geometry
_PHI = (1 + np.sqrt(5)) / 2
_ICO_VERTS_RAW = np.array([
    [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, 1], [-_PHI, 0, -1],
], dtype=float)
_ICO_VERTS = _ICO_VERTS_RAW / np.linalg.norm(_ICO_VERTS_RAW, axis=1, keepdims=True)

# Precompute edges and faces
_ICO_EDGES = []
for _i in range(12):
    for _j in range(_i + 1, 12):
        _d = np.linalg.norm(_ICO_VERTS[_i] - _ICO_VERTS[_j])
        if 1.0 < _d < 2.0:
            _ICO_EDGES.append((_i, _j))

_ICO_FACES = []
for _i in range(12):
    for _j in range(_i + 1, 12):
        for _k in range(_j + 1, 12):
            _d1 = np.linalg.norm(_ICO_VERTS[_i] - _ICO_VERTS[_j])
            _d2 = np.linalg.norm(_ICO_VERTS[_j] - _ICO_VERTS[_k])
            _d3 = np.linalg.norm(_ICO_VERTS[_i] - _ICO_VERTS[_k])
            if 1.0 < _d1 < 2.0 and 1.0 < _d2 < 2.0 and 1.0 < _d3 < 2.0:
                _ICO_FACES.append((_i, _j, _k))

_ICO_EDGE_MIDS = []
for _i, _j in _ICO_EDGES:
    _mid = (_ICO_VERTS[_i] + _ICO_VERTS[_j]) / 2.0
    _mid = _mid / np.linalg.norm(_mid)
    _ICO_EDGE_MIDS.append(_mid)

_ICO_FACE_CENTERS = []
for _i, _j, _k in _ICO_FACES:
    _center = (_ICO_VERTS[_i] + _ICO_VERTS[_j] + _ICO_VERTS[_k]) / 3.0
    _center = _center / np.linalg.norm(_center)
    _ICO_FACE_CENTERS.append(_center)


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in _ICO_VERTS:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    for mid in _ICO_EDGE_MIDS:
        points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    for center in _ICO_FACE_CENTERS:
        points.append(center * r3)
    if len(points) >= n:
        return np.array(points[:n])

    r4 = spacing * np.sqrt(8.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r4)

    return np.array(points[:n])


def _mackay_icosahedron(n, spacing=0.72):
    """Generate a Mackay icosahedron with up to n atoms."""
    if n <= 0:
        return np.empty((0, 3))
    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    # First shell: 12 vertices
    for v in _ICO_VERTS:
        points.append(v * spacing)
    if len(points) >= n:
        return np.array(points[:n])

    # Second shell: 30 edge midpoints at radius sqrt(3)*spacing
    r2 = spacing * np.sqrt(3.0)
    for mid in _ICO_EDGE_MIDS:
        points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    # Third shell: 20 face centers at radius sqrt(5)*spacing
    r3 = spacing * np.sqrt(5.0)
    for center in _ICO_FACE_CENTERS:
        points.append(center * r3)
    if len(points) >= n:
        return np.array(points[:n])

    # Fourth shell: 60 additional points at radius sqrt(8)*spacing
    r4 = spacing * np.sqrt(8.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r4)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42, min_dist_frac=0.7):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_d = min_dist_frac * spacing
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempt += 1
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
        else:
            base = np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _deformed_structure(base, kind, rng, n):
    """Create a slightly deformed version of a base structure."""
    pos = base.copy()
    scale = 0.08
    pos = pos + rng.normal(0, scale, pos.shape)
    return pos


def _soft_sc_energy_gradient(x, n, n_s, m_s, c_s):
    x = x.reshape(n, 3)
    i_idx, j_idx = np.triu_indices(n, k=1)

    diff = x[i_idx] - x[j_idx]
    r2 = np.sum(diff * diff, axis=1)
    r2 = np.maximum(r2, 1e-10)
    r = np.sqrt(r2)

    r_inv_n = r ** (-n_s)
    E_rep = np.sum(r_inv_n)

    r_inv_m = r ** (-m_s)
    S = np.zeros(n)
    np.add.at(S, i_idx, r_inv_m)
    np.add.at(S, j_idx, r_inv_m)

    sqrt_S = np.sqrt(np.maximum(S, 1e-30))
    E_att = -c_s * np.sum(sqrt_S)
    E = E_rep + E_att

    grad = np.zeros((n, 3))

    coeff_rep = -n_s * r ** (-n_s - 2)
    forces_rep = coeff_rep[:, np.newaxis] * diff
    np.add.at(grad, i_idx, forces_rep)
    np.add.at(grad, j_idx, -forces_rep)

    inv_2sqrt_S = np.zeros(n)
    mask = S > 1e-20
    inv_2sqrt_S[mask] = 1.0 / (2.0 * np.sqrt(S[mask]))

    coeff_att_base = c_s * m_s * r ** (-m_s - 2)
    weights = coeff_att_base * (inv_2sqrt_S[i_idx] + inv_2sqrt_S[j_idx])

    forces_att = weights[:, np.newaxis] * diff
    np.add.at(grad, i_idx, forces_att)
    np.add.at(grad, j_idx, -forces_att)

    return E, grad


def _try_relax(pos, potential, kind):
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def _multi_start_relax(pos, potential, kind, rng, n_starts=3, time_limit=8.0):
    """Try multiple relaxed variants from a single starting structure."""
    start = time.time()
    best_e, best_p = _try_relax(pos, potential, kind)
    
    for s in range(n_starts - 1):
        if time.time() > start + time_limit:
            break
        scale = 0.05 + 0.03 * s
        perturbed = pos + rng.normal(0, scale, pos.shape)
        e, p = _try_relax(perturbed, potential, kind)
        if e < best_e:
            best_e, best_p = e, p
    
    return best_e, best_p


def _soft_relax(pos, kind, time_limit=4.0):
    start = time.time()
    n = len(pos)
    pos = pos.copy()

    sp = _SOFT_PARAMS.get(kind, {'n': 6.0, 'm': 3.0, 'c_scale': 0.5})
    n_s = sp['n']
    m_s = sp['m']
    _, _, c_true = _KIND_PARAMS[kind]
    c_s = c_true * sp['c_scale']

    def obj(x):
        if time.time() - start > time_limit:
            raise TimeoutError()
        E, g = _soft_sc_energy_gradient(x, n, n_s, m_s, c_s)
        return E, g.reshape(-1)

    try:
        E0, g0 = obj(pos.flatten())
        if not np.isfinite(E0):
            return np.inf, None

        res = minimize(
            fun=obj,
            x0=pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 600}
        )

        if np.isfinite(res.fun):
            relaxed = res.x.reshape(n, 3)
            if _validate_positions(relaxed):
                return res.fun, relaxed
    except (TimeoutError, Exception):
        pass
    return np.inf, None


def _basin_hopping_step(pos, kind, rng, time_limit=3.0):
    """Perform a single basin hopping step: perturb, soft relax, true relax."""
    n = len(pos)
    scale = 0.1
    perturbed = pos + rng.normal(0, scale, pos.shape)
    
    soft_e, soft_pos = _soft_relax(perturbed, kind, time_limit=time_limit)
    if soft_pos is not None and _validate_positions(soft_pos):
        true_e, true_pos = _try_relax(soft_pos, 'sutton_chen', kind)
        if true_pos is not None:
            return true_e, true_pos
    return np.inf, None


def _high_precision_polish(pos, potential, kind, time_limit=10.0):
    """Final high-precision polish with tighter tolerances."""
    start = time.time()
    try:
        if not _validate_positions(pos):
            return np.inf, None
        
        # Use the standard relax but potentially with more iterations via multiple passes
        best_e, best_p = _try_relax(pos, potential, kind)
        
        # Try a few more perturbations around the polished structure
        rng = np.random.default_rng(12345)
        for trial in range(5):
            if time.time() - start > time_limit:
                break
            scale = 0.03
            perturbed = best_p + rng.normal(0, scale, best_p.shape)
            e, p = _try_relax(perturbed, potential, kind)
            if e < best_e:
                best_e, best_p = e, p
        
        # One more direct relax on the best found
        if time.time() - start < time_limit - 2:
            e2, p2 = _try_relax(best_p, potential, kind)
            if e2 < best_e:
                best_e, best_p = e2, p2
        
        if best_p is not None and _validate_positions(best_p):
            return best_e, best_p
    except Exception:
        pass
    return np.inf, None


def construct(instance):
    start_time = time.time()
    deadline = start_time + 285.0

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    rng = np.random.default_rng(42)

    # Generate diverse structural seeds
    candidates = []
    candidates.append(_fcc_cluster(n, spacing))
    candidates.append(_fcc_cluster(n, spacing * 0.9))
    candidates.append(_fcc_cluster(n, spacing * 1.1))
    candidates.append(_hcp_cluster(n, spacing))
    candidates.append(_hcp_cluster(n, spacing * 0.9))
    candidates.append(_hcp_cluster(n, spacing * 1.1))
    candidates.append(_bcc_cluster(n, spacing))
    candidates.append(_bcc_cluster(n, spacing * 1.05))
    candidates.append(_icosahedral_cluster(n, spacing))
    candidates.append(_icosahedral_cluster(n, spacing * 1.1))
    candidates.append(_icosahedral_cluster(n, spacing * 0.9))
    candidates.append(_mackay_icosahedron(n, spacing))
    candidates.append(_mackay_icosahedron(n, spacing * 0.95))

    # Random seeds
    for seed in range(1, 15):
        if time.time() > deadline - 20:
            break
        for md_frac in [0.65, 0.75, 0.85]:
            if time.time() > deadline - 20:
                break
            pos = _random_compact(n, spacing, seed=seed * 100 + int(md_frac * 100), min_dist_frac=md_frac)
            if _validate_positions(pos):
                candidates.append(pos)

    best_energy = np.inf
    best_positions = None

    # Phase 1: Relax all candidates with multi-start
    for pos in candidates:
        if time.time() > deadline - 15:
            break

        # Try direct relaxation first
        true_e, true_pos = _try_relax(pos, potential, kind)
        if true_e < best_energy:
            best_energy = true_e
            best_positions = true_pos.copy()

        # Also try soft-relax path if time permits
        if time.time() < deadline - 15:
            soft_e, soft_pos = _soft_relax(pos, kind, time_limit=3.0)
            if soft_pos is not None and _validate_positions(soft_pos):
                se, sp = _try_relax(soft_pos, potential, kind)
                if se < best_energy:
                    best_energy = se
                    best_positions = sp.copy()

    # Phase 2: Multi-start refinement from best and diverse structures
    if best_positions is not None and time.time() < deadline - 10:
        e, p = _multi_start_relax(best_positions, potential, kind, rng, n_starts=5, time_limit=12.0)
        if e < best_energy:
            best_energy = e
            best_positions = p.copy()

    # Phase 3: Basin hopping from best configuration with adaptive scaling
    if best_positions is not None and time.time() < deadline - 8:
        bh_trials = 0
        # Adaptive perturbation scales: start small, increase if stuck
        scales = [0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
        scale_idx = 0
        while time.time() < deadline - 8 and bh_trials < 15:
            scale = scales[min(scale_idx, len(scales) - 1)]
            perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
            
            soft_e, soft_pos = _soft_relax(perturbed, kind, time_limit=2.5)
            if soft_pos is not None and _validate_positions(soft_pos):
                true_e, true_pos = _try_relax(soft_pos, 'sutton_chen', kind)
                if true_pos is not None and true_e < best_energy:
                    best_energy = true_e
                    best_positions = true_pos.copy()
                    scale_idx = max(0, scale_idx - 1)  # Reset to smaller scale on success
                else:
                    scale_idx = min(scale_idx + 1, len(scales) - 1)  # Increase scale on failure
            else:
                scale_idx = min(scale_idx + 1, len(scales) - 1)
            bh_trials += 1

    # Phase 4: Polish with perturbations
    if best_positions is not None and time.time() < deadline - 5:
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            for scale in [0.04, 0.08, 0.12, 0.18]:
                if time.time() > deadline - 5:
                    break
                perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
                e, p = _try_relax(perturbed, potential, kind)
                if e < best_energy:
                    best_energy = e
                    best_positions = p.copy()

    # Phase 5: High-precision polish
    if best_positions is not None and time.time() < deadline - 3:
        e_polish, p_polish = _high_precision_polish(best_positions, potential, kind, time_limit=8.0)
        if p_polish is not None and e_polish < best_energy:
            best_energy = e_polish
            best_positions = p_polish.copy()

    # Final polish
    if best_positions is not None and time.time() < deadline - 2:
        try:
            e2, p2 = relax(best_positions, potential, kind)
            if np.isfinite(e2) and _validate_positions(p2) and e2 < best_energy:
                best_energy = e2
                best_positions = p2.copy()
        except Exception:
            pass

    if best_positions is None:
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos

    return best_positions
# EVOLVE-BLOCK-END
