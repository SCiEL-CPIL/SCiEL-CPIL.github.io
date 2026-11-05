"""
Strategy: Basin hopping with simulated annealing acceptance + diverse structured seeds.
Uses icosahedral, FCC, BCC, HCP templates plus random compact clusters as seeds.
Each basin hop performs a local L-BFGS-B minimization, then accepts/rejects
via Metropolis criterion. Multiple independent basin-hopping runs in parallel
across the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_GLOBAL_START_TIME = None
_DEADLINE = 322.0  # safety margin before 330s kill


def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)


def _morse_energy(positions, rho):
    """Compute Morse potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)


def _morse_gradient(positions, rho):
    """Compute gradient of Morse potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    iu = np.triu_indices(n, 1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff ** 2, axis=1))
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    x = rho * (1.0 - r_safe)
    e = np.exp(x)
    dEdr = -2.0 * rho * e * (e - 1.0)
    factor = dEdr / r_safe
    factor = np.where(mask, factor, 0.0)
    np.add.at(grad, iu[0], -factor[:, np.newaxis] * diff)
    np.add.at(grad, iu[1], factor[:, np.newaxis] * diff)
    return grad


def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    if n < 2:
        return np.inf
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))


def _fix_min_distance(positions, min_dist=0.5):
    """Nudge atoms that violate minimum distance constraint."""
    n = len(positions)
    for _ in range(20):
        if _min_distance(positions) >= min_dist:
            break
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        if dist[i, j] < min_dist:
            direction = positions[i] - positions[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            push = (min_dist - dist[i, j]) / 2.0 + 0.01
            positions[i] += direction * push
            positions[j] -= direction * push
    return positions


def _generate_icosahedron(n, spacing=1.0):
    """Generate n atoms in icosahedral shell structure."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0]) * spacing * 0.5

    atoms = [np.zeros(3)]  # center
    # First shell: 12 icosahedral vertices
    for v in ico_verts:
        atoms.append(v.copy())
    # Second shell: 30 edge midpoints + 20 face centers
    # Edge midpoints
    edges = [(0,1),(0,2),(0,4),(0,7),(1,3),(1,5),(2,3),(2,8),
             (3,9),(4,5),(4,6),(4,10),(5,7),(5,10),(6,7),(6,8),
             (7,9),(8,9),(8,11),(9,11),(10,11),(10,6),(11,4),
             (0,5),(1,6),(2,7),(3,8),(4,9),(5,8)]
    for a, b in edges:
        mid = (ico_verts[a] + ico_verts[b]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid = mid / mid_norm * np.linalg.norm(ico_verts[0]) * 1.0
            atoms.append(mid)
    # If we need more, add a third shell (truncated icosahedron-like)
    while len(atoms) < n:
        # Add points on a sphere of radius 2x
        angle = math.acos(1 - 2.0 * len(atoms) / n)
        phi = 2.0 * math.pi * len(atoms)
        r = 2.0 * spacing * 0.5
        p = np.array([r * math.sin(angle) * math.cos(phi),
                      r * math.sin(angle) * math.sin(phi),
                      r * math.cos(angle)])
        atoms.append(p)
    return np.array(atoms[:n])


def _generate_fcc(n, spacing=1.0):
    """Generate n atoms in FCC-like arrangement."""
    atoms = []
    a = spacing * 2.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]


def _generate_bcc(n, spacing=1.0):
    """Generate n atoms in BCC-like arrangement."""
    atoms = []
    a = spacing * 2.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]


def _generate_hcp(n, spacing=1.0):
    """Generate n atoms in HCP-like arrangement."""
    atoms = []
    a = spacing * 2.0
    c = a * math.sqrt(2.0 / 3.0)
    for k in range(-5, 6):
        for i in range(-5, 6):
            for j in range(-5, 6):
                if k % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = (j * math.sqrt(3) / 2.0) * a
                else:
                    x = (i + 0.5 * (j % 2) + 1.0/3.0) * a
                    y = (j * math.sqrt(3) / 2.0 + 1.0/3.0) * a
                z = k * c
                atoms.append([x, y, z])
    atoms = np.array(atoms)
    if len(atoms) >= n:
        dists = np.sum(atoms ** 2, axis=1)
        idx = np.argsort(dists)[:n]
        return atoms[idx]
    else:
        while len(atoms) < n:
            atoms = np.vstack([atoms, np.random.randn(3) * 0.5])
        return atoms[:n]


def _random_compact(n, seed=None):
    """Generate a random compact cluster with minimum separation."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.7
    points = []
    max_attempts = n * 100
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        attempt += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)


def _local_minimize(positions, rho, max_iter=300, time_limit=10.0):
    """Refine positions using L-BFGS-B with time limit."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)

    def jac(x):
        pos = x.reshape(-1, 3)
        g = _morse_gradient(pos, rho)
        return g.flatten()

    x0 = positions.flatten()
    start = time.time()

    def callback(xk):
        return time.time() - start > time_limit

    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-12,
                                   'gtol': 1e-10, 'callback': callback})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy(positions, rho)


def _basin_hop(positions, rho, n_steps, step_size, T0, T1, rng, time_limit):
    """Basin hopping with Metropolis acceptance."""
    current_pos = positions.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    start_time = time.time()

    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        if _get_remaining_time() < 5.0:
            break

        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        proposed_pos = current_pos + perturbation
        proposed_pos = _fix_min_distance(proposed_pos, 0.5)

        # Local minimization
        min_pos, min_energy = _local_minimize(proposed_pos, rho, max_iter=200,
                                              time_limit=min(8.0, time_limit * 0.3))

        # Metropolis acceptance
        T = T0 * (T1 / T0) ** (step / max(n_steps, 1))
        delta_e = min_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            current_pos = min_pos
            current_energy = min_energy

        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()

    return best_pos, best_energy


def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()

    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance["potential"]

    remaining = _get_remaining_time()
    if remaining < 10.0:
        # Emergency: return a quick structured cluster
        pos = _generate_fcc(n, 1.0)
        pos = _fix_min_distance(pos, 0.5)
        pos, _ = _local_minimize(pos, rho, max_iter=100, time_limit=remaining - 2)
        return pos

    # Generate diverse starting points
    starts = []
    starts.append(_generate_icosahedron(n, 1.0))
    starts.append(_generate_fcc(n, 1.0))
    starts.append(_generate_bcc(n, 1.0))
    starts.append(_generate_hcp(n, 1.0))

    for seed in [n, n + 1, n + 2, n + 3, n + 4, n + 5]:
        starts.append(_random_compact(n, seed=seed))

    # Jittered versions
    for base in [_generate_fcc(n), _generate_bcc(n), _generate_icosahedron(n)]:
        starts.append(base + np.random.randn(*base.shape) * 0.08)

    best_energy = np.inf
    best_positions = None

    # Phase 1: Quick local minimization of all starting points
    time_per_quick = max(1.0, min(5.0, remaining / (len(starts) * 2)))
    for i, start_pos in enumerate(starts):
        remaining = _get_remaining_time()
        if remaining < 5.0:
            break
        pos = _fix_min_distance(start_pos.copy(), 0.5)
        min_pos, min_e = _local_minimize(pos, rho, max_iter=200, time_limit=time_per_quick)
        if min_e < best_energy:
            best_energy = min_e
            best_positions = min_pos

    # Phase 2: Basin hopping from the best few starting points
    remaining = _get_remaining_time()
    if remaining > 15.0 and best_positions is not None:
        # Determine number of basin hopping runs
        bh_time = remaining * 0.6
        n_bh_runs = max(2, min(6, int(bh_time / 8.0)))
        bh_time_per_run = bh_time / n_bh_runs

        # Use top starting points for basin hopping
        top_starts = []
        if best_positions is not None:
            top_starts.append(best_positions)
        top_starts.append(_fix_min_distance(_generate_fcc(n).copy(), 0.5))
        top_starts.append(_fix_min_distance(_generate_icosahedron(n).copy(), 0.5))

        for run in range(n_bh_runs):
            remaining = _get_remaining_time()
            if remaining < 5.0:
                break
            start_idx = run % len(top_starts)
            init_pos = top_starts[start_idx].copy()
            rng = np.random.default_rng(seed=n * 100 + run)

            # Adaptive parameters
            step_size = 0.3 + 0.1 * (run % 3)
            T0 = max(1.0, abs(best_energy) * 0.05)
            T1 = T0 * 0.01

            bh_pos, bh_e = _basin_hop(init_pos, rho,
                                       n_steps=max(10, int(bh_time_per_run / 2.0)),
                                       step_size=step_size, T0=T0, T1=T1,
                                       rng=rng, time_limit=bh_time_per_run)
            if bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_pos

    # Phase 3: Final polish with high-precision local minimization
    remaining = _get_remaining_time()
    if remaining > 5.0 and best_positions is not None:
        final_pos, final_e = _local_minimize(best_positions, rho,
                                             max_iter=500,
                                             time_limit=min(remaining - 2, 20.0))
        if final_e < best_energy:
            best_energy = final_e
            best_positions = final_pos

        # Also try the relax function
        remaining = _get_remaining_time()
        if remaining > 8.0:
            try:
                e2, pos2 = relax(best_positions.copy(), potential, rho)
                if e2 < best_energy:
                    best_energy = e2
                    best_positions = pos2
            except Exception:
                pass

    if best_positions is None:
        best_positions = _fix_min_distance(_generate_fcc(n).copy(), 0.5)

    # Final validity check
    best_positions = _fix_min_distance(best_positions, 0.5)

    return best_positions
# EVOLVE-BLOCK-END
