"""Sutton-Chen cluster global minima: diverse lattice seeds + adaptive basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline: 330s wall-clock limit, use 290s as our deadline
_DEADLINE = time.time() + 290.0


def _check_deadline():
    return time.time() > _DEADLINE


def _sc_energy(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-10)
    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    return float(0.5 * np.sum(r_inv_n) - c * np.sum(np.sqrt(r_inv_m)))


def _sc_gradient(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient (vectorized)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-10)

    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    e = 0.5 * np.sum(r_inv_n) - c * np.sum(np.sqrt(r_inv_m))

    # Pair gradient: -0.5*n_exp * sum_j r_ij^(-n_exp-2) * (pos_i - pos_j)
    pair_weight = d ** (-n_exp - 2)
    pair_grad = -0.5 * n_exp * np.sum(pair_weight[:, :, np.newaxis] * diff, axis=1)

    # Cohesive gradient: c*m_exp * sum_j r_ij^(-m_exp-2) * (pos_i - pos_j) / (2*sqrt(S_i))
    S_i = np.maximum(r_inv_m, 1e-30)
    coh_weight = d ** (-m_exp - 2)
    coh_sum = np.sum(coh_weight[:, :, np.newaxis] * diff, axis=1)
    coh_grad = c * m_exp * coh_sum / (2.0 * np.sqrt(S_i)[:, np.newaxis])

    return float(e), pair_grad + coh_grad


def _local_relax(pos, n_exp, m_exp, c, maxiter=400):
    """Local relaxation using L-BFGS-B with analytic gradient."""
    if _check_deadline():
        return _sc_energy(pos, n_exp, m_exp, c), pos

    def objective(x):
        e, g = _sc_gradient(x, n_exp, m_exp, c)
        return e, g

    try:
        res = minimize(objective, pos, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-7, 'ftol': 1e-10})
        e, _ = _sc_gradient(res.x, n_exp, m_exp, c)
        return e, res.x
    except Exception:
        return _sc_energy(pos, n_exp, m_exp, c), pos


def _generate_fcc_seed(n, spacing=0.72):
    """Generate FCC-like cluster: nearest n atoms to origin on FCC lattice."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    max_idx = int(np.ceil(n ** (1 / 3))) + 3
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p)
    points = np.array(points) * spacing
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp_seed(n, spacing=0.72):
    """Generate HCP-like cluster."""
    a = spacing
    c_hcp = a * np.sqrt(2.0 / 3.0)
    points = []
    # Multiple hexagonal layers
    for layer in range(6):
        z = layer * c_hcp
        is_a = (layer % 2 == 0)
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3) / 2 * j
                if not is_a:
                    x += a / 2
                    y += a / (2 * np.sqrt(3))
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_icosahedral_seed(n, spacing=0.72):
    """Generate icosahedral-like cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts * spacing / np.sqrt(1 + phi ** 2)

    points = [np.zeros(3)]
    for v in verts:
        points.append(v.copy())

    # Second shell: 30 atoms at edge midpoints (icosahedral)
    # Find edges (pairs of vertices with distance ~ edge length)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < 0.1:
                mid = (verts[i] + verts[j]) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-10:
                    points.append(mid * (spacing * 1.2 / mid_norm))

    # Third shell: 20 face centers
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - edge_len) < 0.1 and abs(d2 - edge_len) < 0.1 and abs(d3 - edge_len) < 0.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc_norm = np.linalg.norm(fc)
                    if fc_norm > 1e-10:
                        points.append(fc * (spacing * 1.4 / fc_norm))

    # Pad with random points if needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.normal(0, spacing * 0.6, 3)
        points.append(p)

    return np.array(points[:n])


def _generate_random_compact(n, spacing=0.72, seed=0):
    """Generate random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1 / 3) * 0.85
    points = []
    min_dist = 0.5 * spacing
    for _ in range(n):
        placed = False
        for _ in range(100):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
                    placed = True
                    break
        if not placed:
            # Relax constraint
            for _ in range(50):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    points.append(p)
                    break
            else:
                points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)


def _basin_hopping(pos, n_exp, m_exp, c, n_iter=40, step_size=0.25,
                   temp_start=0.5, temp_end=0.01, seed=0):
    """Adaptive basin hopping with cooling schedule."""
    if _check_deadline():
        return _sc_energy(pos, n_exp, m_exp, c), pos

    best_e, best_pos = _local_relax(pos, n_exp, m_exp, c)
    current_e, current_pos = best_e, best_pos.copy()
    rng = np.random.default_rng(seed)

    for i in range(n_iter):
        if _check_deadline():
            break
        # Cooling schedule
        temp = temp_start * (temp_end / temp_start) ** (i / max(n_iter - 1, 1))
        # Adaptive step: start larger, decrease
        step = step_size * (1.0 + 0.5 * (temp / temp_start))

        # Perturbation
        delta = rng.normal(0, step, size=current_pos.shape)
        trial_pos = current_pos + delta

        # Local relax
        trial_e, trial_pos = _local_relax(trial_pos, n_exp, m_exp, c, maxiter=300)

        # Metropolis acceptance
        if trial_e < current_e:
            current_e, current_pos = trial_e, trial_pos
        else:
            accept_prob = np.exp(-(trial_e - current_e) / max(temp, 1e-10))
            if rng.random() < accept_prob:
                current_e, current_pos = trial_e, trial_pos

        if current_e < best_e:
            best_e, best_pos = current_e, current_pos.copy()

    return best_e, best_pos


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster using diverse seeds + adaptive basin hopping."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]

    best_energy = np.inf
    best_pos = None

    # Generate diverse initial structures
    seeds = []
    seeds.append(_generate_fcc_seed(size))
    seeds.append(_generate_hcp_seed(size))
    seeds.append(_generate_icosahedral_seed(size))
    seeds.append(_generate_random_compact(size, seed=42))
    seeds.append(_generate_random_compact(size, seed=123))

    # First pass: relax each seed
    for i, seed_pos in enumerate(seeds):
        if _check_deadline():
            break
        e, pos = _local_relax(seed_pos, n_exp, m_exp, c, maxiter=500)
        if e < best_energy:
            best_energy, best_pos = e, pos.copy()

    # Second pass: basin hopping from the best relaxed seeds
    # Sort seeds by their relaxed energy and do BH on top ones
    relaxed_seeds = []
    for seed_pos in seeds:
        if _check_deadline():
            break
        e, pos = _local_relax(seed_pos, n_exp, m_exp, c, maxiter=300)
        relaxed_seeds.append((e, pos))

    relaxed_seeds.sort(key=lambda x: x[0])

    # Basin hopping on top 3 seeds
    for i, (e, pos) in enumerate(relaxed_seeds[:3]):
        if _check_deadline():
            break
        bh_e, bh_pos = _basin_hopping(pos, n_exp, m_exp, c,
                                       n_iter=35, step_size=0.25,
                                       temp_start=0.4, temp_end=0.01,
                                       seed=1000 + i * 7)
        if bh_e < best_energy:
            best_energy, best_pos = bh_e, bh_pos.copy()

    # If we still have time, do one more round of BH with different parameters
    if not _check_deadline() and best_pos is not None:
        bh_e, bh_pos = _basin_hopping(best_pos, n_exp, m_exp, c,
                                       n_iter=25, step_size=0.15,
                                       temp_start=0.2, temp_end=0.005,
                                       seed=9999)
        if bh_e < best_energy:
            best_energy, best_pos = bh_e, bh_pos.copy()

    # Final relaxation
    if best_pos is not None:
        final_e, final_pos = _local_relax(best_pos, n_exp, m_exp, c, maxiter=500)
        return final_pos

    # Fallback
    return _generate_fcc_seed(size)
# EVOLVE-BLOCK-END
