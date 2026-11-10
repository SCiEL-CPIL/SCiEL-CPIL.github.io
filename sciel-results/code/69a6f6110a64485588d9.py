"""Morse cluster global minima via parallel basin hopping with diverse crystalline/icosahedral starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute Morse energy directly."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    return 0.5 * np.sum(pair_energy)


def _make_fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster with n atoms."""
    positions = []
    # Generate FCC lattice points
    a = spacing
    max_coord = 10
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC basis
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + bx, j + by, k + bz]) * a
                    if len(positions) < n:
                        positions.append(p)
    return np.array(positions[:n])


def _make_icosahedral_cluster(n, spacing=1.0):
    """Generate an icosahedral cluster with n atoms (nested shells)."""
    positions = []
    # Center
    positions.append(np.zeros(3))
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    # Normalize and scale
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis] * spacing
    
    # Add icosahedron shell
    for v in ico_verts:
        if len(positions) < n:
            positions.append(v)
    
    # If we need more, add a second shell (dodecahedron-like)
    if len(positions) < n:
        # Generate dodecahedron vertices
        phi2 = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi2
        dode_verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, inv_phi, phi2], [0, -inv_phi, phi2], [0, inv_phi, -phi2], [0, -inv_phi, -phi2],
            [inv_phi, phi2, 0], [-inv_phi, phi2, 0], [inv_phi, -phi2, 0], [-inv_phi, -phi2, 0],
            [phi2, 0, inv_phi], [phi2, 0, -inv_phi], [-phi2, 0, inv_phi], [-phi2, 0, -inv_phi],
        ])
        norms = np.linalg.norm(dode_verts, axis=1)
        dode_verts = dode_verts / norms[:, np.newaxis] * spacing * 1.8
        for v in dode_verts:
            if len(positions) < n:
                positions.append(v)
    
    # If still need more, add random points on a sphere
    while len(positions) < n:
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        r = spacing * 2.5
        p = r * np.array([sin_theta * np.cos(theta), sin_theta * np.sin(theta), cos_theta])
        positions.append(p)
    
    return np.array(positions[:n])


def _make_bcc_cluster(n, spacing=1.0):
    """Generate a BCC-like cluster with n atoms."""
    positions = []
    a = spacing
    max_coord = 10
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # BCC basis
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + bx, j + by, k + bz]) * a
                    if len(positions) < n:
                        positions.append(p)
    return np.array(positions[:n])


def _make_random_cluster(n, spacing=1.0, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    positions = []
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in positions):
            positions.append(p)
    return np.array(positions)


def _basin_hop(positions, rho, max_steps=50, step_size=0.1, rng=None):
    """Perform basin hopping: random kick + local relaxation."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(positions.copy(), 'morse', rho)
    
    for step in range(max_steps):
        # Random perturbation
        kick = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + kick
        
        # Local relaxation
        new_energy, new_pos = relax(new_pos, 'morse', rho)
        
        # Accept if lower energy
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            # Decrease step size slowly
            step_size *= 0.98
    
    return current_energy, current_pos


def _worker(args):
    """Worker function for parallel basin hopping."""
    start_positions, rho, max_steps, step_size, seed = args
    rng = np.random.default_rng(seed)
    return _basin_hop(start_positions, rho, max_steps, step_size, rng)


def construct(instance):
    """Find low-energy Morse cluster configurations via parallel basin hopping."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Generate diverse starting configurations
    starts = []
    starts.append(_make_fcc_cluster(n))
    starts.append(_make_icosahedral_cluster(n))
    starts.append(_make_bcc_cluster(n))
    starts.append(_make_random_cluster(n, seed=1))
    starts.append(_make_random_cluster(n, seed=2))
    starts.append(_make_random_cluster(n, seed=3))
    
    # For each start, do basin hopping
    best_energy = np.inf
    best_positions = None
    
    # Sequential first pass to get a good initial guess
    for i, start in enumerate(starts[:3]):
        if time.time() > deadline:
            break
        energy, positions = _basin_hop(start, rho, max_steps=30, step_size=0.15, rng=np.random.default_rng(i))
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Parallel basin hopping with more steps
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_positions is not None:
        # Use the best so far as a new starting point for more exploration
        max_workers = min(4, concurrent.futures.cpu_count())
        
        tasks = []
        for i in range(max_workers):
            # Perturb the best configuration
            rng = np.random.default_rng(100 + i)
            perturbed = best_positions + rng.normal(0, 0.2, size=best_positions.shape)
            tasks.append((perturbed, rho, 40, 0.12, 200 + i))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                if time.time() > deadline:
                    break
                try:
                    energy, positions = future.result(timeout=1)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    pass
    
    # Final sequential refinement if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_positions is not None:
        energy, positions = _basin_hop(best_positions, rho, max_steps=20, step_size=0.05, rng=np.random.default_rng(999))
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        # Check and fix any violations
        for _ in range(10):
            diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) >= 0.5:
                break
            # Nudge atoms that are too close
            min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_dist_idx
            direction = best_positions[i] - best_positions[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0, 0])
            direction /= norm
            push = (0.5 - np.min(dist)) * 0.5 + 0.01
            best_positions[i] += direction * push
            best_positions[j] -= direction * push
    
    return best_positions
# EVOLVE-BLOCK-END
