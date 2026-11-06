"""Dzugutov cluster global minima: L-BFGS-B local opt + simulated annealing + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 290.0  # 10s safety margin before 300s kill
_THREAD_LOCK = threading.Lock()
_best_results = {}


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a set of positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    E = 0.0
    mask1 = r_ij < a
    mask2 = (r_ij >= a) & (r_ij < b)
    if np.any(mask1):
        r1 = r_ij[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    if np.any(mask2):
        r2 = r_ij[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    return float(E)


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron."""
    t = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _random_rotation(rng):
    """Generate a random 3D rotation matrix."""
    # QR decomposition of random Gaussian matrix
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return Q


def _generate_icosahedral_seed(n, rng):
    """Generate an icosahedral subunit assembly seed."""
    positions = [np.zeros(3)]
    
    # First shell: 12 atoms (icosahedron)
    ico = _icosahedron_vertices()
    rot = _random_rotation(rng)
    r1 = 1.05
    for v in ico:
        positions.append((rot @ v) * r1)
    
    remaining = n - len(positions)
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: icosahedron rotated and scaled
    r2 = 1.85
    rot2 = _random_rotation(rng)
    # Use icosahedron vertices for second shell
    ico2 = _icosahedron_vertices()
    # Add additional points to fill up to needed count
    second_shell_points = []
    for v in ico2:
        second_shell_points.append((rot2 @ v) * r2)
    
    # If we need more, add more icosahedral shells or fibonacci points
    if remaining > len(second_shell_points):
        # Add more from a slightly larger icosahedron
        r3 = 2.3
        rot3 = _random_rotation(rng)
        for v in ico2:
            second_shell_points.append((rot3 @ v) * r3)
    
    # Take only as many as needed
    for p in second_shell_points[:remaining]:
        positions.append(p)
    
    # If still short, add random points on sphere
    while len(positions) < n:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = r2 + rng.uniform(0, 0.3)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        positions.append(p)
    
    return np.array(positions[:n])


def _generate_compact_random(n, rng, radius_scale=1.0):
    """Generate a compact random cluster with minimum distance constraint."""
    radius = 1.0 * n ** (1.0 / 3.0) * radius_scale
    points = [np.zeros(3)]
    attempts = 0
    while len(points) < n and attempts < n * 100:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_dist = min(np.linalg.norm(p - q) for q in points) if points else np.inf
        if min_dist >= 0.6:
            points.append(p)
    
    # If we couldn't place all, add more aggressively
    while len(points) < n:
        p = rng.uniform(-radius * 1.2, radius * 1.2, 3)
        points.append(p)
    
    return np.array(points[:n])


def _local_minimize_lbfgs(pos, potential, parameter, rng):
    """Use L-BFGS-B with gradient for local minimization."""
    try:
        # Flatten positions
        x0 = pos.flatten()
        
        def energy_and_grad(x):
            pos_2d = x.reshape(-1, 3)
            # Use the provided gradient function
            try:
                energy, grad = dzugutov_energy_gradient(pos_2d)
                return energy, grad.flatten()
            except Exception:
                # Fallback to numerical gradient
                e = _dzugutov_energy(pos_2d)
                # Simple numerical gradient
                grad = np.zeros_like(x)
                step = 1e-5
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += step
                    e_plus = _dzugutov_energy(x_plus.reshape(-1, 3))
                    grad[i] = (e_plus - e) / step
                return e, grad
        
        result = minimize(
            energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _dzugutov_energy(pos):
            return result.fun, result.x.reshape(-1, 3)
        else:
            return _dzugutov_energy(pos), pos
    except Exception:
        return _dzugutov_energy(pos), pos


def _simulated_annealing(start_pos, potential, parameter, rng, t_initial=2.0, t_final=0.01, 
                          cooling=0.995, max_iter=100, kick_scale=0.2):
    """Simulated annealing to escape local minima."""
    current_pos = start_pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = t_initial
    
    for i in range(max_iter):
        if _remaining() < 5:
            break
        
        # Propose a move
        kick = rng.normal(0, kick_scale, size=current_pos.shape)
        proposed_pos = current_pos + kick
        
        # Compute proposed energy
        proposed_energy = _dzugutov_energy(proposed_pos)
        
        # Metropolis criterion
        delta_e = proposed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_pos = proposed_pos
            current_energy = proposed_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool down
        T *= cooling
    
    # Final local minimization
    final_energy, final_pos = _local_minimize_lbfgs(best_pos, potential, parameter, rng)
    if final_energy < best_energy:
        best_energy = final_energy
        best_pos = final_pos
    
    return best_energy, best_pos


def _basin_hopping_lbfgs(start_pos, potential, parameter, rng, max_iter=20, kick_scale=0.25):
    """Basin hopping using L-BFGS-B for local minimization."""
    best_pos = start_pos.copy()
    best_energy = _dzugutov_energy(best_pos)
    
    for i in range(max_iter):
        if _remaining() < 10:
            break
        
        # Kick
        kick = rng.normal(0, kick_scale, size=best_pos.shape)
        kicked = best_pos + kick
        
        # Local minimize with L-BFGS-B
        try:
            energy, relaxed = _local_minimize_lbfgs(kicked, potential, parameter, rng)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except Exception:
            continue
    
    return best_energy, best_pos


def _worker_basin_hopping(seed_idx, start_pos, potential, parameter, n, rng_seed):
    """Worker function for parallel basin hopping."""
    rng = np.random.default_rng(rng_seed)
    try:
        # First local minimize
        e0, pos0 = _local_minimize_lbfgs(start_pos, potential, parameter, rng)
        
        # Basin hopping
        e, pos = _basin_hopping_lbfgs(pos0, potential, parameter, rng, max_iter=15, kick_scale=0.25)
        
        return seed_idx, e, pos
    except Exception as e:
        return seed_idx, np.inf, start_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using L-BFGS-B + SA + parallel basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if size in _best_results:
        return _best_results[size]
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse seeds
    seeds = []
    
    # Icosahedral seeds (multiple rotations)
    for i in range(3):
        seeds.append(_generate_icosahedral_seed(size, rng))
    
    # Compact random seeds with different scales
    for scale in [0.8, 1.0, 1.2]:
        seeds.append(_generate_compact_random(size, rng, radius_scale=scale))
    
    # Relax all seeds with L-BFGS-B and find best
    best_energy = np.inf
    best_pos = seeds[0]
    
    for i, seed in enumerate(seeds):
        if _remaining() < 20:
            break
        try:
            energy, pos = _local_minimize_lbfgs(seed, potential, parameter, rng)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except Exception:
            continue
    
    # Simulated annealing from best position
    if _remaining() > 40:
        try:
            sa_energy, sa_pos = _simulated_annealing(
                best_pos, potential, parameter, rng,
                t_initial=3.0, t_final=0.05, cooling=0.99, max_iter=50, kick_scale=0.2
            )
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pos = sa_pos
        except Exception:
            pass
    
    # Parallel basin hopping from multiple starting points
    if _remaining() > 30:
        # Create multiple starting points for parallel BH
        bh_starts = [best_pos.copy()]
        for i in range(3):
            kick = rng.normal(0, 0.3, size=best_pos.shape)
            bh_starts.append(best_pos + kick)
        
        # Run parallel basin hopping
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx, start in enumerate(bh_starts):
                future = executor.submit(
                    _worker_basin_hopping, idx, start, potential, parameter, size,
                    size * 1000 + 42 + idx * 100
                )
                futures.append(future)
            
            for future in as_completed(futures):
                if _remaining() < 15:
                    break
                try:
                    idx, e, pos = future.result(timeout=30)
                    if e < best_energy:
                        best_energy = e
                        best_pos = pos
                except Exception:
                    continue
    
    # Final L-BFGS-B refinement
    if _remaining() > 5:
        try:
            final_energy, final_pos = _local_minimize_lbfgs(best_pos, potential, parameter, rng)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass
    
    _best_results[size] = best_pos
    return best_pos
# EVOLVE-BLOCK-END
