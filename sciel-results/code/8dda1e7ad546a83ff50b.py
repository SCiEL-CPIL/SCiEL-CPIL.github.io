"""Dzugutov cluster global minima: icosahedral symmetry manifold + basin hopping with parallel restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 295.0  # safety margin under 300s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    e = np.zeros_like(r_ij)
    
    # Second term: B exp(d/(r-b)) for r < b
    mask_b = r_ij < _b
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        r_b_safe = np.maximum(r_b, 1e-10)
        e[mask_b] += _B * np.exp(_d / (r_b_safe - _b))
    
    # First term: A(r^-m - B)exp(c/(r-a)) for r < a
    mask_a = r_ij < _a
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        r_a_safe = np.maximum(r_a, 1e-10)
        e[mask_a] += _A * (r_a_safe ** (-_m) - _B) * np.exp(_c / (r_a_safe - _a))
    
    return float(np.sum(e))


def _icosahedron_vertices():
    """Return unit icosahedron vertices and edge/face data."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges
    edge_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.0:
                edge_pairs.append((i, j))
    
    # Edge midpoints (unit directions)
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            edge_dirs.append(mid / mid_norm)
    edge_dirs = np.array(edge_dirs)
    
    # Face centers (unit directions)
    face_dirs = []
    for i, j, k in combinations(range(12), 3):
        d12 = np.linalg.norm(verts[i] - verts[j])
        d13 = np.linalg.norm(verts[i] - verts[k])
        d23 = np.linalg.norm(verts[j] - verts[k])
        if d12 < 1.0 and d13 < 1.0 and d23 < 1.0:
            fc = (verts[i] + verts[j] + verts[k]) / 3.0
            fc_norm = np.linalg.norm(fc)
            if fc_norm > 1e-10:
                face_dirs.append(fc / fc_norm)
    face_dirs = np.array(face_dirs)
    
    return verts, edge_dirs, face_dirs


_VERTS, _EDGE_DIRS, _FACE_DIRS = _icosahedron_vertices()


def _build_icosahedral(n, r1, r2_edge, r2_face, twist=0.0, rng=None):
    """Build an icosahedral cluster with adjustable shell radii and twist.
    
    Parameters:
    - r1: radius of shell 1 (12 atoms)
    - r2_edge: radius of shell 2 edge-midpoint atoms (30 atoms)
    - r2_face: radius of shell 2 face-center atoms (20 atoms)
    - twist: angular twist applied to shell 2 relative to shell 1
    - rng: random number generator for shuffling
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    atoms = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices
    if n > 1:
        shell1 = _VERTS * r1
        atoms.extend(shell1.tolist())
    
    # Shell 2: edge midpoints + face centers
    if n > 13:
        # Apply twist to shell 2 directions
        if twist != 0.0:
            # Rotation around z-axis by twist angle
            ct = np.cos(twist)
            st = np.sin(twist)
            rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
            edge_dirs_twisted = _EDGE_DIRS @ rot.T
            face_dirs_twisted = _FACE_DIRS @ rot.T
        else:
            edge_dirs_twisted = _EDGE_DIRS
            face_dirs_twisted = _FACE_DIRS
        
        shell2_edge = edge_dirs_twisted * r2_edge
        shell2_face = face_dirs_twisted * r2_face
        
        remaining = n - len(atoms)
        shell2_atoms = list(shell2_edge) + list(shell2_face)
        rng.shuffle(shell2_atoms)
        atoms.extend(shell2_atoms[:remaining])
    
    # If still need more atoms, add random
    if len(atoms) < n:
        remaining = n - len(atoms)
        for _ in range(remaining):
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            r = r1 * rng.uniform(1.5, 2.2)
            atoms.append(dir_vec * r)
    
    return np.array(atoms[:n])


def _optimize_icosahedral_params(n, seed, time_budget):
    """Optimize icosahedral symmetry parameters (r1, r2_edge, r2_face, twist) via Nelder-Mead."""
    from scipy.optimize import minimize
    
    rng = np.random.default_rng(seed)
    
    # Initial guess: standard Mackay with NN ~ 1.05
    r1_init = 1.05
    r2_edge_init = r1_init * np.sqrt(3.0) / 2.0
    r2_face_init = r1_init * np.sqrt(3.0) / (2.0 * np.sqrt(5.0))
    twist_init = 0.0
    
    def objective(params):
        r1, r2e, r2f, twist = params
        # Ensure positive radii
        r1 = max(r1, 0.5)
        r2e = max(r2e, 0.5)
        r2f = max(r2f, 0.3)
        try:
            pos = _build_icosahedral(n, r1, r2e, r2f, twist, rng)
            # Check minimum distance constraint
            if n > 1:
                dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
                iu = np.triu_indices(n, k=1)
                if np.min(dists[iu]) < 0.49:
                    return 1e6
            e = _energy(pos)
            return e
        except Exception:
            return 1e6
    
    x0 = np.array([r1_init, r2_edge_init, r2_face_init, twist_init])
    
    # Use Nelder-Mead with reasonable bounds
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_params = result.x
    best_r1 = max(best_params[0], 0.5)
    best_r2e = max(best_params[1], 0.5)
    best_r2f = max(best_params[2], 0.3)
    best_twist = best_params[3]
    
    pos = _build_icosahedral(n, best_r1, best_r2e, best_r2f, best_twist, rng)
    e = _energy(pos)
    
    return e, pos, (best_r1, best_r2e, best_r2f, best_twist)


def _basin_hopping(positions, potential, parameter, max_iter=60, step_size=0.3, seed=0):
    """Adaptive basin hopping: random perturbations + local relaxation with cooling."""
    rng = np.random.default_rng(seed)
    pos = np.array(positions, dtype=float)
    
    # Initial relaxation
    energy, pos = relax(pos, potential, parameter)
    best_energy, best_pos = energy, pos.copy()
    
    n_atoms = pos.shape[0]
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Adaptive step size: start large, cool down
        progress = it / max(max_iter, 1)
        current_step = step_size * (1.0 - 0.6 * progress)
        
        # Random perturbation
        noise = rng.normal(0, current_step, pos.shape)
        new_pos = pos + noise
        
        # Relax
        energy, new_pos = relax(new_pos, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = new_pos.copy()
        
        # Metropolis acceptance
        dE = energy - best_energy
        T = 0.5 * (1.0 - 0.8 * progress)
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            pos = new_pos
    
    return best_energy, best_pos


def _random_compact(n, rng):
    """Generate a compact random cluster with minimum distance constraint."""
    pos = np.zeros((n, 3))
    pos[0] = np.zeros(3)
    
    for i in range(1, n):
        for attempt in range(500):
            p = rng.normal(0, 1.2, 3)
            if i > 0:
                dists = np.linalg.norm(pos[:i] - p, axis=1)
                if np.min(dists) >= 0.5:
                    pos[i] = p
                    break
        else:
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            pos[i] = dir_vec * rng.uniform(1.5, 2.5)
    
    return pos


def _single_search(n, potential, parameter, seed, time_budget):
    """Run a single search thread with time budget."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    strategy = seed % 4
    
    if strategy == 0:
        # Optimize icosahedral parameters, then basin hop
        try:
            e0, pos0, params = _optimize_icosahedral_params(n, seed, time_budget)
            if _remaining() < 5.0:
                return e0, pos0
            best_energy, best_pos = _basin_hopping(pos0, potential, parameter,
                                                    max_iter=50, step_size=0.25, seed=seed)
            return best_energy, best_pos
        except Exception:
            # Fallback to standard icosahedron
            pos0 = _build_icosahedral(n, 1.05, 0.91, 0.46, 0.0, rng)
            best_energy, best_pos = _basin_hopping(pos0, potential, parameter,
                                                    max_iter=60, step_size=0.30, seed=seed)
            return best_energy, best_pos
    
    elif strategy == 1:
        # Perturbed icosahedron with larger twist
        twist = rng.uniform(-0.3, 0.3)
        r1 = rng.uniform(1.0, 1.1)
        r2e = r1 * np.sqrt(3.0) / 2.0 * rng.uniform(0.95, 1.05)
        r2f = r1 * np.sqrt(3.0) / (2.0 * np.sqrt(5.0)) * rng.uniform(0.95, 1.05)
        pos0 = _build_icosahedral(n, r1, r2e, r2f, twist, rng)
        pos0 += rng.normal(0, 0.1, pos0.shape)
        best_energy, best_pos = _basin_hopping(pos0, potential, parameter,
                                                max_iter=60, step_size=0.35, seed=seed)
        return best_energy, best_pos
    
    elif strategy == 2:
        # Compact random cluster
        pos0 = _random_compact(n, rng)
        best_energy, best_pos = _basin_hopping(pos0, potential, parameter,
                                                max_iter=60, step_size=0.30, seed=seed)
        return best_energy, best_pos
    
    else:
        # Standard icosahedron with moderate perturbation
        pos0 = _build_icosahedral(n, 1.05, 0.91, 0.46, 0.0, rng)
        pos0 += rng.normal(0, 0.08, pos0.shape)
        best_energy, best_pos = _basin_hopping(pos0, potential, parameter,
                                                max_iter=60, step_size=0.28, seed=seed)
        return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure using icosahedral symmetry manifold and basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance
    
    # Time budget for this instance
    time_budget = min(_remaining() - 10.0, 45.0)
    if time_budget < 5.0:
        time_budget = 5.0
    
    # Run multiple parallel searches with different seeds
    n_threads = 4
    seeds = list(range(n_threads))
    
    best_energy = float('inf')
    best_pos = None
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for seed in seeds:
            future = executor.submit(_single_search, n, potential, parameter, seed, time_budget)
            futures.append(future)
        
        for future in as_completed(futures, timeout=max(_remaining() - 2.0, 5.0)):
            try:
                energy, pos = future.result(timeout=1.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue
    
    # If no valid result, use a simple relaxed icosahedron
    if best_pos is None:
        init_pos = _build_icosahedral(n, 1.05, 0.91, 0.46, 0.0)
        best_energy, best_pos = relax(init_pos, potential, parameter)
    
    return best_pos
# EVOLVE-BLOCK-END
