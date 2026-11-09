"""Morse cluster global minima search with basin-hopping and multi-threaded search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: evaluator kills after 330s wall-clock.
_DEADLINE = time.time() + 315.0  # 15s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    pair_e = e * (e - 2.0)
    return 0.5 * np.sum(pair_e)

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    # dE/dr_ij = rho * (2*e - 2) * e = 2*rho*e*(e-1)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    dEdr = 2.0 * rho * e * (e - 1.0)
    dEdr[np.diag_indices(n)] = 0.0
    # Weighted sum
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] > 1e-10:
                grad[i] += dEdr[i, j] * diff[i, j] / dist[i, j]
    return grad

def _local_minimize(pos, rho, max_steps=500, tol=1e-8):
    """Simple gradient descent with adaptive step size."""
    pos = pos.copy()
    n = len(pos)
    energy = _morse_energy(pos, rho)
    step = 0.01
    
    for step_num in range(max_steps):
        grad = _morse_gradient(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        new_pos = pos - step * grad
        new_energy = _morse_energy(new_pos, rho)
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.5
            if step < 1e-6:
                break
        
        # Check minimum separation
        if n >= 2:
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.49:
                break
    
    return energy, pos

def _basin_hopping(pos, rho, n_iterations=20, perturb_scale=0.5, rng=None):
    """Basin hopping: local minimize, perturb, repeat."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    n = len(pos)
    current_energy, current_pos = _local_minimize(pos, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(n_iterations):
        # Perturb
        perturbation = rng.normal(0, perturb_scale, pos.shape)
        perturbed = current_pos + perturbation
        
        # Local minimize
        new_energy, new_pos = _local_minimize(perturbed, rho)
        
        # Accept if better or with Metropolis criterion
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            # Metropolis acceptance (simplified: always accept if close)
            if new_energy - current_energy < 0.1:
                current_energy = new_energy
                current_pos = new_pos
    
    return best_energy, best_pos

def _generate_fcc_cluster(n, spacing=1.0, rng=None):
    """Generate compact FCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Generate points in a cubic lattice
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for i in range(-side, side):
        for j in range(-side, side):
            for k in range(-side, side):
                if len(points) >= n:
                    break
                points.append([i * spacing, j * spacing, k * spacing])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.02, points.shape)
    
    return points

def _generate_hcp_cluster(n, spacing=1.0, rng=None):
    """Generate HCP-like cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    points = []
    a = spacing
    c = a * np.sqrt(2.0 / 3.0)
    
    # Generate layers
    for layer in range(20):
        z = layer * c
        # AB stacking
        offset = [0, 0] if layer % 2 == 0 else [a/2, a*np.sqrt(3)/6]
        
        for i in range(-5, 6):
            for j in range(-5, 6):
                if len(points) >= n:
                    break
                x = i * a + offset[0]
                y = j * a * np.sqrt(3) / 2 + offset[1]
                points.append([x, y, z])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.02, points.shape)
    
    return points

def _generate_icosahedral(n, radius=1.0, rng=None):
    """Generate points on a spherical shell."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.column_stack([x, y, z])

def _search_one_instance(size, rho, time_budget, rng_seed):
    """Search for minimum energy configuration of one instance."""
    rng = np.random.default_rng(rng_seed)
    
    # Generate diverse initial configurations
    radius = size ** (1.0/3.0) * 0.8
    
    initial_configs = [
        _generate_fcc_cluster(size, 1.0, rng),
        _generate_hcp_cluster(size, 1.0, rng),
        _generate_icosahedral(size, radius, rng),
    ]
    
    # Add random compact configurations
    for scale in [0.7, 1.0, 1.3]:
        r = radius * scale
        points = []
        min_dist = 0.55
        attempts = 0
        while len(points) < size and attempts < size * 50:
            p = rng.uniform(-r, r, 3)
            if np.linalg.norm(p) <= r:
                if points:
                    dists = np.linalg.norm(np.array(points) - p, axis=1)
                    if np.min(dists) >= min_dist:
                        points.append(p)
                else:
                    points.append(p)
            attempts += 1
        while len(points) < size:
            p = rng.uniform(-r, r, 3)
            points.append(p)
        initial_configs.append(np.array(points))
    
    best_energy = np.inf
    best_positions = None
    deadline = time.time() + time_budget
    
    for config in initial_configs:
        if time.time() > deadline - 2.0:
            break
        
        try:
            # Use basin hopping
            energy, pos = _basin_hopping(config, rho, n_iterations=15, perturb_scale=0.4, rng=rng)
            
            # Verify minimum separation
            if len(pos) >= 2:
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.49:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = pos.copy()
        except Exception:
            continue
    
    # If no valid result from basin hopping, try simple relax
    if best_positions is None:
        for config in initial_configs:
            if time.time() > deadline - 1.0:
                break
            try:
                energy, relaxed_pos = relax(config, 'morse', rho)
                if len(relaxed_pos) >= 2:
                    diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
                    dist = np.sqrt(np.sum(diff * diff, axis=2))
                    np.fill_diagonal(dist, np.inf)
                    if np.min(dist) < 0.49:
                        continue
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
            except Exception:
                continue
    
    if best_positions is None:
        best_positions = _generate_icosahedral(size, radius, rng)
    
    return best_positions, best_energy

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    time_for_instance = max(10.0, min(50.0, _time_left() / 6.0))
    
    rng_seed = size * 1000 + int(parameter)
    
    # Use multi-threaded search with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            seed = rng_seed + i * 12345
            futures.append(executor.submit(_search_one_instance, size, parameter, time_for_instance / 4.0, seed))
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=time_for_instance)
                results.append(result)
            except Exception:
                continue
    
    # Pick the best result
    best_positions = None
    best_energy = np.inf
    for pos, energy in results:
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    if best_positions is None:
        # Fallback
        radius = size ** (1.0/3.0) * 0.8
        best_positions = _generate_icosahedral(size, radius, np.random.default_rng(rng_seed))
    
    return best_positions
# EVOLVE-BLOCK-END
