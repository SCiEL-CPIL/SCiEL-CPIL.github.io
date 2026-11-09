"""Dzugutov cluster global minima search.
Combines parallel multi-basin exploration (Mackay + physics-driven self-assembly
seeds) with basin-hopping Metropolis refinement driven by the robust relax API."""

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

# Global deadline guard (evaluator kills at 330s wall-clock)
TOTAL_TIME_LIMIT = 305.0
_start_time = None
_best_results = {}


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _energy_gradient(positions):
    """Compute Dzugutov energy and gradient (vectorized fallback)."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)

    V = np.zeros((n, n))
    dVdr = np.zeros((n, n))
    mask = r < A_CUT
    if np.any(mask):
        rm = r[mask]
        rm_safe = np.where(rm > 1e-8, rm, 1e-8)
        rm_safe = np.clip(rm_safe, 1e-8, A_CUT - 1e-6)
        Vm = (A_POT * (rm_safe ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (rm_safe - A_CUT))
        dVm = A_POT * (-M_EXP * rm_safe ** (-M_EXP - 1)
                       - C_EXP / (rm_safe - A_CUT) ** 2 * (rm_safe ** (-M_EXP) - B_POT)) \
              * np.exp(C_EXP / (rm_safe - A_CUT))
        V[mask] = Vm
        dVdr[mask] = dVm
    mask2 = (r >= A_CUT) & (r < B_CUT)
    if np.any(mask2):
        rm = r[mask2]
        rm_safe = np.where(rm > 1e-8, rm, 1e-8)
        rm_safe = np.clip(rm_safe, 1e-8, B_CUT - 1e-6)
        Vm = B_POT * np.exp(D_EXP / (rm_safe - B_CUT))
        dVm = B_POT * np.exp(D_EXP / (rm_safe - B_CUT)) * (-D_EXP / (rm_safe - B_CUT) ** 2)
        V[mask2] = Vm
        dVdr[mask2] = dVm

    energy = 0.5 * np.sum(V)
    factor = dVdr / r
    grad = -np.sum(factor[:, None] * diff, axis=1)
    return energy, grad


def _enforce_min_distance(positions, rng, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = np.asarray(positions, dtype=float).copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    for _ in range(30):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = rng.normal(0, 1, 3)
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _make_mackay_icosahedron(n, rng):
    """Mackay icosahedron-like polytetrahedral seed."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    shells = [np.zeros((1, 3))]
    remaining = n - 1

    if remaining > 0:
        shell1 = v * 1.0
        take1 = min(12, remaining)
        shells.append(shell1[:take1])
        remaining -= take1

    if remaining > 0:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if d < 1.1:
                    mid = (v[i] + v[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edges.append(mid)
        unique_edges = []
        for e in edges:
            if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
                unique_edges.append(e)
        r2 = np.sqrt(1 + 2 / phi)
        shell2 = np.array(unique_edges) * r2
        take2 = min(len(shell2), remaining)
        shells.append(shell2[:take2])
        remaining -= take2

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
            fc = (v[f[0]] + v[f[1]] + v[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
        r3 = np.sqrt(1 + 2 / phi + 2 / (phi * phi))
        shell3 = np.array(face_centers) * r3
        take3 = min(len(shell3), remaining)
        shells.append(shell3[:take3])
        remaining -= take3

    if remaining > 0:
        r4 = 2.8
        for _ in range(remaining):
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * r4
            shells.append(p.reshape(1, 3))

    positions = np.vstack(shells)
    if len(positions) > n:
        radii = np.linalg.norm(positions, axis=1)
        indices = np.argsort(radii)[:n]
        positions = positions[indices]
    return positions


def _make_fcc_cluster(n, spacing, rng):
    """Compact FCC-like cluster."""
    positions = [np.zeros(3)]
    remaining = n - 1
    a = spacing * 0.9
    lattice_pts = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                lattice_pts.append(np.array([ix, iy, iz]) * a)
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    lattice_pts.append(np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a)
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]
    lattice_pts = lattice_pts[dists > 0.1]
    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])
    while len(positions) < n:
        positions.append(rng.uniform(-4 * spacing, 4 * spacing, 3))
    return np.array(positions[:n])


def _make_random_compact(n, spacing, rng):
    """Random compact cluster."""
    positions = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    for _ in range(5000):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)
    while len(positions) < n:
        positions.append(rng.uniform(-radius, radius, 3))
    return np.array(positions[:n])


def _self_assemble(n, rng, time_budget):
    """Physics-driven self-assembly: random start + Langevin annealing to relax."""
    radius = 1.1 * n ** (1.0 / 3.0) * 0.9
    pos = rng.uniform(-radius, radius, (n, 3))
    pos = _enforce_min_distance(pos, rng)
    vel = np.zeros_like(pos)
    dt = 0.002
    T = 1.5
    T_end = 0.05
    steps = 1500
    t_start = time.time()
    for s in range(steps):
        if time.time() - t_start > time_budget:
            break
        try:
            _, grad = _energy_gradient(pos)
        except Exception:
            break
        # Langevin: v = (v - grad*dt) * damping + sqrt(2*T*dt)*noise
        vel = (vel - grad * dt) * 0.9 + np.sqrt(2 * T * dt) * rng.normal(size=pos.shape)
        pos = pos + vel * dt
        T = max(T_end, T * 0.999)
        if s % 150 == 0:
            pos = _enforce_min_distance(pos, rng)
    return pos


def _relax_positions(positions):
    """Relax via the relax API with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None:
            return float(energy), np.asarray(pos, dtype=float)
    except Exception:
        pass

    def objective_with_grad(x):
        p = x.reshape(-1, 3)
        try:
            e, g = _energy_gradient(p)
            return e, g.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    best_energy, best_x = 1e10, x0.copy()
    try:
        result = minimize(objective_with_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 30, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    return best_energy, best_x.reshape(-1, 3)


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping: kick, relax, Metropolis accept, track best."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    if current_energy is None or np.isinf(current_energy):
        return None, None
    best_energy, best_pos = current_energy, current_pos.copy()
    t_start = time.time()
    T = 1.0
    for _ in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break
        kicked = current_pos + rng.normal(0, kick_scale, positions.shape)
        kicked = _enforce_min_distance(kicked, rng)
        new_energy, new_pos = _relax_positions(kicked)
        if new_energy is None:
            continue
        delta_e = new_energy - current_energy
        accept = delta_e < 0 or (rng.random() < np.exp(-delta_e / T))
        if accept:
            current_energy, current_pos = new_energy, new_pos
            if new_energy < best_energy:
                best_energy, best_pos = new_energy, new_pos.copy()
        T *= 0.95
    return best_energy, best_pos


def _worker_basin_hopping(args):
    positions, n_kicks, kick_scale, time_budget, seed = args
    try:
        return _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed)
    except Exception:
        return None, None


def _parallel_basin_hopping(tasks, n_workers=4):
    if len(tasks) == 0:
        return []
    n_workers = min(n_workers, len(tasks), mp.cpu_count() or 1)
    if n_workers <= 1:
        return [_worker_basin_hopping(t) for t in tasks]
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_basin_hopping, tasks)
        return results
    except Exception:
        return [_worker_basin_hopping(t) for t in tasks]


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]
    if n in _best_results:
        return _best_results[n]

    rng = np.random.default_rng(42 + n)
    spacing = 1.1
    base_budget = max(10.0, min(_remaining() * 0.9, (TOTAL_TIME_LIMIT / 6.0) * 1.5))

    # Build diverse seeds: Mackay, self-assembly (physics-driven), FCC, random, perturbed
    seeds = []
    seeds.append(_make_mackay_icosahedron(n, rng))
    # Physics-driven self-assembly seeds (structure emerges from the potential)
    for sa_seed in range(2):
        rng_sa = np.random.default_rng(1000 + n + sa_seed * 7)
        seeds.append(_self_assemble(n, rng_sa, time_budget=min(6.0, base_budget * 0.2)))
    seeds.append(_make_fcc_cluster(n, spacing, rng))
    for ro in range(2):
        rng_r = np.random.default_rng(42 + n + ro * 100)
        seeds.append(_make_random_compact(n, spacing, rng_r))
    for scale in [0.95, 1.05]:
        seeds.append(seeds[0] * scale + rng.normal(0, 0.05, seeds[0].shape))

    processed = [_enforce_min_distance(s, rng) for s in seeds]

    # Parallel basin hopping over seeds
    bh_budget = max(4.0, base_budget * 0.75)
    n_kicks = max(6, int(bh_budget / 2.0))
    tasks = [(s, n_kicks, 0.3, bh_budget, 5000 + n + i) for i, s in enumerate(processed)]
    results = _parallel_basin_hopping(tasks, n_workers=4)

    best_energy, best_positions = np.inf, None
    for energy, pos in results:
        if energy is not None and pos is not None and energy < best_energy:
            best_energy, best_positions = energy, pos

    # If basin hopping produced nothing, relax seeds directly
    if best_positions is None:
        for s in processed:
            if _remaining() < 3:
                break
            e, p = _relax_positions(s)
            if e is not None and e < best_energy:
                best_energy, best_positions = e, p

    # Final fallback
    if best_positions is None:
        best_positions = processed[0] if processed else rng.uniform(-3, 3, (n, 3))

    final_pos = _enforce_min_distance(best_positions, rng)
    _best_results[n] = final_pos
    return final_pos
