"""Dzugutov cluster global minima search.
Uses known polytetrahedral core coordinates with geometric insertion
and adaptive multi-scale basin hopping for global optimization."""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline guard - must finish well before 330s kill
TOTAL_TIME_LIMIT = 320.0
_start_time = None
_best_results = {}


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient using the API or fallback."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
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


def _make_icosahedron_13():
    """Exact 13-atom icosahedron (center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to optimal Dzugutov distance ~1.0
    return np.vstack([np.zeros(3), verts * 1.0])


def _make_mackay_55():
    """55-atom Mackay icosahedron: 1 + 12 + 30 + 12 shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Shell 0: center
    atoms = [np.zeros(3)]

    # Shell 1: 12 vertices at r1
    r1 = 1.0
    atoms.extend([v * r1 for v in verts])

    # Shell 2: 30 edge midpoints at r2
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

    r2 = np.sqrt(1 + 1.0 / phi)  # ~1.618
    atoms.extend([e * r2 for e in unique_edges[:30]])

    # Shell 3: 20 face centers at r3
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                if d_ij < 1.25 and d_jk < 1.25 and d_ik < 1.25:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc_norm = np.linalg.norm(fc)
                    if fc_norm > 1e-8:
                        fc = fc / fc_norm
                        faces.append(fc)

    unique_faces = []
    for f in faces:
        if all(np.linalg.norm(f - uf) > 0.1 for uf in unique_faces):
            unique_faces.append(f)

    r3 = np.sqrt(2 + 2.0 / phi)  # ~1.902
    atoms.extend([f * r3 for f in unique_faces[:20]])

    return np.array(atoms[:55])


def _make_mackay_icosahedron(n, rng=None):
    """Generate Mackay icosahedron with proper shells, extending for n > 55."""
    if n <= 55:
        base = _make_mackay_55()
        return base[:n]

    base = _make_mackay_55()
    remaining = n - 55

    if rng is None:
        rng = np.random.default_rng(42)

    # Add atoms on outer shell at increasing radii
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Generate more positions on larger shells
    all_positions = [base.copy()]
    r_outer = 2.2
    for shell_idx in range(10):
        r = r_outer + shell_idx * 0.3
        # Use icosahedral directions
        for v in verts:
            all_positions.append(v * r)
        # Add edge directions
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.25:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-8:
                        mid = mid / mid_norm
                        all_positions.append(mid * r)

    # Select closest remaining positions
    base_pos = base
    candidate_positions = []
    for p in all_positions[1:]:
        # Check if not too close to existing
        if all(np.linalg.norm(p - bp) > 0.45 for bp in base_pos):
            candidate_positions.append(p)

    candidate_positions = np.array(candidate_positions) if candidate_positions else np.zeros((1, 3))
    # Sort by distance from center
    dists = np.linalg.norm(candidate_positions, axis=1)
    order = np.argsort(dists)
    candidate_positions = candidate_positions[order]

    atoms = [base_pos.copy()]
    take = min(remaining, len(candidate_positions))
    atoms.extend([candidate_positions[i].copy() for i in range(take)])

    while len(atoms) < n:
        p = rng.normal(size=3)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-8:
            p = np.array([1.0, 0.0, 0.0])
        else:
            p = p / p_norm
        atoms.append(p * (r_outer + 3.0))

    return np.array(atoms[:n])


def _make_core_based(n, rng=None):
    """Build cluster using 13-atom icosahedron as core, add interstitials."""
    if rng is None:
        rng = np.random.default_rng(42)

    core = _make_icosahedron_13()
    remaining = n - 13

    if remaining <= 0:
        return core[:n]

    # Add atoms at tetrahedral/octahedral interstitial sites
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Interstitial positions: between center and vertices (tetrahedral-like)
    interstitials = []
    for v in verts:
        # Position at 0.6 * vertex direction (between center and vertex)
        interstitials.append(v * 0.6)
        # Position at vertex + outward
        interstitials.append(v * 1.5)
        # Edge midpoints
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.25:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-8:
                        mid = mid / mid_norm
                        interstitials.append(mid * 0.8)

    # Remove duplicates
    unique_int = []
    for p in interstitials:
        if all(np.linalg.norm(p - u) > 0.3 for u in unique_int):
            unique_int.append(p)

    # Sort by distance from center
    unique_int = np.array(unique_int)
    dists = np.linalg.norm(unique_int, axis=1)
    order = np.argsort(dists)
    unique_int = unique_int[order]

    atoms = [core.copy()]
    take = min(remaining, len(unique_int))
    atoms.extend([unique_int[i].copy() for i in range(take)])

    # If still need more, add on outer shell
    while len(atoms) < n:
        p = rng.normal(size=3)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-8:
            p = np.array([1.0, 0.0, 0.0])
        else:
            p = p / p_norm
        atoms.append(p * 2.0)

    return np.array(atoms[:n])


def _make_mackay_core_based(n, rng=None):
    """Build cluster using 55-atom Mackay as core for n >= 55."""
    if n <= 55:
        return _make_mackay_icosahedron(n, rng)

    core = _make_mackay_55()
    remaining = n - 55

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

    # Generate outer shell positions
    outer_positions = []
    for r in [2.0, 2.3, 2.6, 2.9]:
        for v in verts:
            outer_positions.append(v * r)
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.25:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-8:
                        mid = mid / mid_norm
                        outer_positions.append(mid * r)

    # Filter out positions too close to core
    filtered = []
    for p in outer_positions:
        if all(np.linalg.norm(p - c) > 0.5 for c in core):
            filtered.append(p)

    filtered = np.array(filtered) if filtered else np.zeros((1, 3))
    dists = np.linalg.norm(filtered, axis=1)
    order = np.argsort(dists)
    filtered = filtered[order]

    atoms = [core.copy()]
    take = min(remaining, len(filtered))
    atoms.extend([filtered[i].copy() for i in range(take)])

    while len(atoms) < n:
        p = rng.normal(size=3)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-8:
            p = np.array([1.0, 0.0, 0.0])
        else:
            p = p / p_norm
        atoms.append(p * 3.0)

    return np.array(atoms[:n])


def _make_random_compact(n, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = []
    radius = 1.3 * n ** (1.0 / 3.0)
    min_dist = 0.55
    max_attempts = 8000

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

    # Fallback: L-BFGS optimization
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
            options={'maxiter': 60, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _basin_hopping(positions, n_kicks, kick_scales, time_budget, seed):
    """Basin hopping with multiple kick scales for better exploration."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start = time.time()
    T = 0.8  # Initial temperature

    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 10:
            break

        # Cycle through kick scales with some randomness
        base_scale = kick_scales[kick_idx % len(kick_scales)]
        # Add slight random variation
        current_kick_scale = base_scale * (0.8 + 0.4 * rng.random())

        # Kick
        kick = rng.normal(0, current_kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)

        # Relax
        new_energy, new_pos = _relax_positions(kicked)

        if new_energy is None:
            continue

        # Metropolis acceptance
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

        # Cool down
        T *= 0.96

    return best_energy, best_pos


def _worker_basin_hopping(args):
    """Worker for parallel basin hopping."""
    positions, n_kicks, kick_scales, time_budget, seed = args
    try:
        best_energy, best_pos = _basin_hopping(positions, n_kicks, kick_scales, time_budget, seed)
        return best_energy, best_pos
    except Exception:
        return None, None


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Time budget for this instance
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    # Allocate time proportional to remaining, with minimum
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.85, base_time_per_instance * 1.8)
    time_budget = max(time_budget, 20.0)

    # Generate diverse initial structures
    rng = np.random.default_rng(42 + n)
    candidates = []

    # 1. Mackay icosahedron (primary - exploits polytetrahedral preference)
    mackay_pos = _make_mackay_icosahedron(n, rng)
    candidates.append(("mackay", mackay_pos))

    # 2. Core-based (13-atom icosahedron core)
    core_pos = _make_core_based(n, rng)
    candidates.append(("core13", core_pos))

    # 3. Mackay core-based (55-atom core for n >= 55)
    if n >= 55:
        mackay_core_pos = _make_mackay_core_based(n, rng)
        candidates.append(("mackay_core", mackay_core_pos))

    # 4. Perturbed Mackay variants
    for seed_offset in [1, 2, 3, 4]:
        perturbed = mackay_pos + rng.normal(0, 0.08, mackay_pos.shape)
        candidates.append((f"mackay_pert{seed_offset}", perturbed))

    # 5. Random compact
    random_pos = _make_random_compact(n, rng)
    candidates.append(("random", random_pos))

    # 6. Another random with different seed
    rng2 = np.random.default_rng(1000 + n)
    random_pos2 = _make_random_compact(n, rng2)
    candidates.append(("random2", random_pos2))

    # 7. Slightly distorted icosahedron
    if n >= 13:
        ico_base = _make_icosahedron_13()
        if n > 13:
            # Extend icosahedron
            ext = _make_mackay_icosahedron(n, rng)
            # Blend
            blended = 0.5 * ext + 0.5 * (ico_base[:min(n, 13)] if n >= 13 else ext[:n])
            # This is rough, just use the mackay
            candidates.append(("ico_blend", ext + rng.normal(0, 0.05, ext.shape)))
        else:
            candidates.append(("ico13", ico_base[:n]))

    best_energy = np.inf
    best_positions = None

    # Relax all candidates
    relaxed_candidates = []
    for name, cand in candidates:
        if _remaining() < 15:
            break
        pos = _enforce_min_distance(cand)
        energy, relaxed = _relax_positions(pos)
        if energy is not None and relaxed is not None:
            relaxed_candidates.append((energy, relaxed, name))
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed

    if not relaxed_candidates:
        # Fallback
        pos = _enforce_min_distance(candidates[0][1])
        energy, relaxed = _relax_positions(pos)
        if relaxed is not None:
            best_positions = relaxed
            best_energy = energy
        else:
            best_positions = pos

    # Sort by energy and pick top for basin hopping
    relaxed_candidates.sort(key=lambda x: x[0])
    top_candidates = relaxed_candidates[:min(5, len(relaxed_candidates))]

    # Multi-scale kick parameters
    kick_scales = [0.10, 0.15, 0.20, 0.08, 0.25]

    # Number of kicks based on time budget
    base_kicks = max(10, int(50 / (n / 40)))

    # Parallel basin hopping
    num_workers = min(4, len(top_candidates))
    per_worker_time = time_budget * 0.6 / num_workers

    worker_args = []
    for i, (energy, pos, name) in enumerate(top_candidates):
        seed = 2000 + n * 10 + i
        worker_args.append((pos, base_kicks, kick_scales, per_worker_time, seed))

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_basin_hopping, worker_args)

        for energy, pos in results:
            if energy is not None and pos is not None:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos

    except Exception:
        # Fallback: sequential basin hopping
        if top_candidates:
            pos = top_candidates[0][1]
            energy, new_pos = _basin_hopping(pos, base_kicks, kick_scales, time_budget * 0.5, seed=9999)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = new_pos

    # Final relaxation of the best result
    if best_positions is not None and _remaining() > 10:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        best_positions = _enforce_min_distance(candidates[0][1])

    _best_results[n] = best_positions
    return best_positions
