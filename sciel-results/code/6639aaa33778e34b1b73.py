"""Dzugutov cluster global minima search.
Combines self-assembly polytetrahedral seeding with threaded basin hopping."""

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline
_TOTAL_START = None
_HARD_LIMIT = 295.0
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_best_results = {}


def _elapsed():
    if _TOTAL_START is None:
        return 0.0
    return time.time() - _TOTAL_START


def _remaining():
    return max(0.0, _HARD_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient."""
    try:
        energy, grad = dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue

            if r < A_CUT:
                V = (A_POT * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
                dVdr = A_POT * (-M_EXP * r ** (-M_EXP - 1) - C_EXP / (r - A_CUT) ** 2 * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
            elif r < B_CUT:
                V = B_POT * np.exp(D_EXP / (r - B_CUT))
                dVdr = B_POT * np.exp(D_EXP / (r - B_CUT)) * (-D_EXP / (r - B_CUT) ** 2)
            else:
                continue

            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec

    return energy, grad


def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(40):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff = diff / diff_norm
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _relax_positions(positions):
    """Relax positions using the relax API, with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None and pos is not None:
            return energy, pos
    except Exception:
        pass

    def objective_with_grad(x):
        pos = x.reshape(-1, 3)
        try:
            energy, grad = _dzugutov_energy_gradient(pos)
            return energy, grad.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    best_energy = 1e10
    best_x = x0.copy()

    try:
        result = minimize(
            objective_with_grad, x0, method='L-BFGS-B',
            options={'maxiter': 80, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _self_assemble(n_atoms, seed, time_budget=10.0):
    """
    Generate a seed via sequential self-assembly:
    start with an icosahedral nucleus, then add atoms one at a time
    at the lowest-energy position, relaxing after each addition.
    This naturally builds polytetrahedral order.
    """
    rng = np.random.default_rng(seed)
    t_start = time.time()

    # Start with center atom
    positions = [np.zeros(3)]

    # Add icosahedral shell (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms

    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        p = v * 1.0 + rng.normal(0, 0.03, 3)
        # Check minimum distance
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)

    # Relax the nucleus
    if len(positions) > 1:
        pos_arr = np.array(positions)
        try:
            energy, pos_arr = relax(pos_arr, 'dzugutov')
            positions = [pos_arr[i] for i in range(len(positions))]
        except Exception:
            pass

    # Sequentially add remaining atoms
    while len(positions) < n_atoms:
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 3:
            break

        pos_arr = np.array(positions)
        centroid = pos_arr.mean(axis=0)
        radius = np.max(np.linalg.norm(pos_arr - centroid, axis=1))

        # Try multiple candidate positions, pick the best
        best_energy = np.inf
        best_pos = None

        n_candidates = 20
        for trial in range(n_candidates):
            # Place new atom on a sphere of radius slightly larger than current
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            new_r = radius * rng.uniform(0.85, 1.25)
            p_new = centroid + new_r * np.array([
                sin_phi * np.cos(theta),
                sin_phi * np.sin(theta),
                cos_phi
            ])

            # Check minimum distance
            if any(np.linalg.norm(p_new - q) < 0.5 for q in positions):
                continue

            trial_positions = np.array(positions + [p_new])
            try:
                e, relaxed = relax(trial_positions, 'dzugutov')
                if e is not None and relaxed is not None and e < best_energy:
                    best_energy = e
                    best_pos = relaxed
            except Exception:
                pass

        if best_pos is not None:
            positions = [best_pos[i] for i in range(len(best_pos))]
        else:
            # Fallback: add at a random position
            p_new = centroid + radius * rng.normal(0, 0.3, 3)
            p_new = _enforce_min_distance(np.array(positions + [p_new]))[-1]
            positions.append(p_new)

    return np.array(positions[:n_atoms])


def _make_mackay_icosahedron(n, rng=None):
    """Generate a Mackay icosahedron cluster with n atoms."""
    if rng is None:
        rng = np.random.default_rng(42)

    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    positions = [np.zeros(3)]
    remaining = n - 1

    # Shell 1: 12 vertices
    r1 = 1.0
    shell1 = verts * r1
    take1 = min(12, remaining)
    positions.extend([p.copy() for p in shell1[:take1]])
    remaining -= take1

    # Shell 2: 30 edge midpoints
    if remaining > 0:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.25:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-8:
                        mid = mid / mid_norm
                        edges.append(mid)

        unique_edges = []
        for e in edges:
            if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
                unique_edges.append(e)

        r2 = np.sqrt(2 * (1 + 1.0 / phi))
        shell2 = np.array(unique_edges) * r2
        take2 = min(len(shell2), remaining)
        positions.extend([p.copy() for p in shell2[:take2]])
        remaining -= take2

    # Shell 3: 20 face centers
    if remaining > 0:
        face_indices = [
            [0, 1, 8], [0, 1, 11], [0, 2, 8], [0, 2, 9],
            [0, 3, 9], [0, 3, 11], [1, 4, 8], [1, 4, 10],
            [1, 5, 10], [1, 5, 11], [2, 4, 8], [2, 4, 9],
            [2, 5, 9], [2, 5, 10], [3, 6, 9], [3, 6, 11],
            [3, 7, 11], [3, 7, 10], [6, 4, 8], [6, 7, 10]
        ]

        face_centers = []
        for f in face_indices:
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            fc_norm = np.linalg.norm(fc)
            if fc_norm > 1e-8:
                fc = fc / fc_norm
                face_centers.append(fc)

        r3 = np.sqrt(3 * (1 + 1.0 / phi))
        shell3 = np.array(face_centers) * r3
        take3 = min(len(shell3), remaining)
        positions.extend([p.copy() for p in shell3[:take3]])
        remaining -= take3

    # Shell 4: random on larger sphere
    if remaining > 0:
        r4 = 2.8
        for _ in range(remaining):
            p = rng.normal(size=3)
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-8:
                p = np.array([1.0, 0.0, 0.0])
            else:
                p = p / p_norm
            positions.append(p * r4)

    if len(positions) > n:
        pos_arr = np.array(positions)
        radii = np.linalg.norm(pos_arr, axis=1)
        indices = np.argsort(radii)[:n]
        positions = [pos_arr[i] for i in indices]

    return np.array(positions)


def _make_fcc_cluster(n, rng=None):
    """Generate a compact FCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = [np.zeros(3)]
    remaining = n - 1

    a = 1.0
    lattice_pts = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pc = np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a
                    lattice_pts.append(pc)

    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]

    nonzero_mask = dists > 0.1
    lattice_pts = lattice_pts[nonzero_mask]

    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])

    while len(positions) < n:
        p = rng.uniform(-4, 4, 3)
        positions.append(p)

    return np.array(positions[:n])


def _make_random_compact(n, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.6
    max_attempts = 5000

    for _ in range(max_attempts):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)

    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)

    return np.array(positions[:n])


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping with Metropolis acceptance."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start = time.time()
    T = 0.4

    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break

        progress = kick_idx / max(1, n_kicks)
        current_kick_scale = kick_scale * (1.0 - 0.5 * progress)

        kick = rng.normal(0, current_kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)

        new_energy, new_pos = _relax_positions(kicked)

        if new_energy is None:
            continue

        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)

        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()

        T *= 0.96

    return best_energy, best_pos


def _worker_basin_hopping(args):
    """Worker for parallel basin hopping."""
    positions, n_kicks, kick_scale, time_budget, seed = args
    try:
        best_energy, best_pos = _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed)
        return best_energy, best_pos
    except Exception:
        return None, None


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _TOTAL_START
    if _TOTAL_START is None:
        _TOTAL_START = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Time budget for this instance
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = _HARD_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 20.0)

    best_energy = np.inf
    best_positions = None

    # Phase 1: Generate diverse initial structures
    rng = np.random.default_rng(42 + n)

    # 1. Self-assembly seeds (primary - these naturally find polytetrahedral order)
    self_assembly_budget = min(time_budget * 0.35, 30.0)
    sa_seeds = []
    n_sa_seeds = 3
    for sa_i in range(n_sa_seeds):
        if _remaining() < 10:
            break
        sa_seed = _self_assemble(n, seed=100 + n * 10 + sa_i, time_budget=self_assembly_budget / n_sa_seeds)
        if sa_seed is not None and len(sa_seed) == n:
            sa_seeds.append(sa_seed)

    # 2. Mackay icosahedron
    mackay_pos = _make_mackay_icosahedron(n, rng)

    # 3. Perturbed Mackay
    perturbed_mackay = mackay_pos + rng.normal(0, 0.08, mackay_pos.shape)

    # 4. FCC-like
    fcc_pos = _make_fcc_cluster(n, rng)

    # 5. Random compact
    random_pos = _make_random_compact(n, rng)

    # Relax all candidates
    candidates = []
    for cand in sa_seeds + [mackay_pos, perturbed_mackay, fcc_pos, random_pos]:
        pos = _enforce_min_distance(cand)
        energy, relaxed = _relax_positions(pos)
        if energy is not None and relaxed is not None:
            candidates.append((energy, relaxed))
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed

    if not candidates:
        pos = _enforce_min_distance(mackay_pos)
        energy, relaxed = _relax_positions(pos)
        if relaxed is not None:
            best_positions = relaxed
            best_energy = energy
        else:
            best_positions = pos

    # Phase 2: Basin hopping from top candidates
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:min(5, len(candidates))]

    # Use threading (max 4 threads)
    num_workers = min(4, len(top_candidates))
    remaining_time = _remaining() * 0.85
    per_worker_time = remaining_time / num_workers

    # Number of kicks depends on time budget
    base_kicks = max(10, int(50 / (n / 40)))
    kick_scale = 0.10 + 0.05 * (n / 60)

    worker_args = []
    for i, (energy, pos) in enumerate(top_candidates[:num_workers]):
        seed = 5000 + n * 10 + i
        worker_args.append((pos, base_kicks, kick_scale, per_worker_time, seed))

    # Run basin hopping in threads
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_basin_hopping, args) for args in worker_args]
            for future in as_completed(futures):
                if _remaining() < 5:
                    break
                try:
                    energy, pos = future.result(timeout=5)
                    if energy is not None and pos is not None:
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = pos
                except Exception:
                    pass
    except Exception:
        # Fallback: sequential basin hopping
        if top_candidates:
            pos = top_candidates[0][1]
            energy, new_pos = _basin_hopping(pos, base_kicks, kick_scale, remaining_time * 0.5, seed=9999)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = new_pos

    # Phase 3: Additional self-assembly + basin hopping if time remains
    if _remaining() > 20 and len(sa_seeds) < 5:
        extra_sa_time = min(_remaining() * 0.4, 25.0)
        extra_sa = _self_assemble(n, seed=2000 + n, time_budget=extra_sa_time)
        if extra_sa is not None and len(extra_sa) == n:
            pos = _enforce_min_distance(extra_sa)
            energy, relaxed = _relax_positions(pos)
            if energy is not None and relaxed is not None:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
                # Basin hop from this new seed
                if _remaining() > 10:
                    bh_energy, bh_pos = _basin_hopping(
                        relaxed, base_kicks // 2, kick_scale * 0.8,
                        _remaining() * 0.5, seed=7000 + n
                    )
                    if bh_energy is not None and bh_energy < best_energy:
                        best_energy = bh_energy
                        best_positions = bh_pos

    # Final relaxation
    if best_positions is not None and _remaining() > 5:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        best_positions = _enforce_min_distance(mackay_pos)

    _best_results[n] = best_positions
    return best_positions
