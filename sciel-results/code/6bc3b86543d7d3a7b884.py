"""Optimized Morse cluster minimizer using structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


# Global time management
TOTAL_TIME_BUDGET = 290.0  # Leave 10s margin for overhead
start_time_global = time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)  # Avoid self-interaction
    exp_term = np.exp(rho * (1 - dist))
    energy = np.sum(exp_term * (exp_term - 2))
    return energy


def _morse_gradient(positions, rho):
    """Compute Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    exp_term = np.exp(rho * (1 - dist))
    # Gradient: dE/dr_ij * dr_ij/dx_i
    # E_ij = exp(rho(1-r))(exp(rho(1-r))-2)
    # dE/dr = exp(rho(1-r)) * (-rho) * (exp(rho(1-r))-2) + exp(rho(1-r)) * exp(rho(1-r)) * (-rho)
    #       = -rho * exp(rho(1-r)) * (exp(rho(1-r))-2 + exp(rho(1-r)))
    #       = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    #       = -2*rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    grad_dist = -2 * rho * exp_term * (exp_term - 1)
    # dr_ij/dx_i = -diff_ij / r_ij
    grad_dist = grad_dist / dist
    np.fill_diagonal(grad_dist, 0)
    # Gradient on position i is sum over j of grad_dist[i,j] * (-diff[i,j])
    # Actually: dE/dx_i = sum_j (dE/dr_ij) * (dr_ij/dx_i) = sum_j grad_dist[i,j] * (-diff[i,j]/r_ij)
    # But we already divided by r_ij, so:
    grad = -np.sum(grad_dist[:, :, np.newaxis] * diff, axis=1)
    return grad


def generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate points in FCC lattice shells
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * spacing
    
    points = []
    max_shell = int(np.ceil(n ** (1/3.0))) + 3
    
    # Generate candidate points in shells
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * spacing + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    # Greedily pick points maintaining minimum distance
    min_dist = 0.5 * spacing
    points = [candidates[0]]
    for p in candidates[1:]:
        if len(points) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    if len(points) < n:
        # Fill remaining with random points
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-max_shell * spacing, max_shell * spacing, 3)
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    return np.array(points[:n])


def generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # HCP basis: ABAB stacking
    a = spacing
    c = a * np.sqrt(8/3)
    
    # Generate HCP lattice points
    candidates = []
    max_shell = int(np.ceil(n ** (1/3.0))) + 3
    
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell // 2, max_shell // 2 + 1):
                # Layer A
                x = a * (i + j/2)
                y = a * (np.sqrt(3)/2) * j
                z = c * k
                candidates.append([x, y, z])
                # Layer B (offset)
                if k % 2 == 1:
                    x2 = a * (i + j/2) + a/2
                    y2 = y + a * np.sqrt(3)/6
                    z2 = z + c/2
                    candidates.append([x2, y2, z2])
    
    candidates = np.array(candidates)
    
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    # Greedily pick points maintaining minimum distance
    min_dist = 0.5 * spacing
    points = [candidates[0]]
    for p in candidates[1:]:
        if len(points) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-max_shell * spacing, max_shell * spacing, 3)
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    return np.array(points[:n])


def generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis] * spacing
    
    # Generate shells
    points = [np.zeros(3)]  # Center
    shell_radii = [spacing, 1.4 * spacing, 1.8 * spacing, 2.2 * spacing]
    
    for r in shell_radii:
        if len(points) >= n:
            break
        # Place atoms on sphere of radius r
        n_on_shell = min(n - len(points), int(2 * np.pi * r ** 2 / (spacing ** 2)))
        for _ in range(n_on_shell):
            if len(points) >= n:
                break
            # Random point on sphere
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(-1, 1)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * r
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # Fill remaining
    if len(points) < n:
        rng = np.random.default_rng(42)
        for _ in range(10000):
            if len(points) >= n:
                break
            p = rng.uniform(-2.5 * spacing, 2.5 * spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    return np.array(points[:n])


def generate_random_cluster(n, spacing=1.0, seed=None):
    """Generate random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = 100000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # Fill remaining with closer packing
    if len(points) < n:
        for _ in range(100000):
            if len(points) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in points):
                    points.append(p)
    
    return np.array(points[:n])


def local_minimize(positions, rho, max_iter=1000):
    """Local minimization using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    result = minimize(
        objective,
        positions.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-6}
    )
    return result.x.reshape(-1, 3), result.fun


def basin_hopping(positions, rho, n_iterations=50, step_size=0.5, time_limit=30):
    """Simple basin hopping algorithm."""
    best_pos = positions.copy()
    best_energy = _morse_energy(best_pos, rho)
    rng = np.random.default_rng(42)
    start = time.time()
    
    for i in range(n_iterations):
        if time.time() - start > time_limit:
            break
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, best_pos.shape)
        new_pos = best_pos + perturbation
        
        # Local minimize
        new_pos, new_energy = local_minimize(new_pos, rho, max_iter=200)
        
        # Check minimum distance constraint
        if np.any(np.linalg.norm(new_pos[:, np.newaxis] - new_pos[np.newaxis, :], axis=2) < 0.5):
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
    
    return best_pos, best_energy


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    # Time budget for this instance
    elapsed = time.time() - start_time_global
    time_remaining = TOTAL_TIME_BUDGET - elapsed
    if time_remaining < 10:
        # Fallback: just return a valid configuration
        positions = generate_fcc_shells(size, spacing=1.0)
        energy, positions = relax(positions, potential, rho)
        return positions
    
    # Generate diverse initial configurations
    initializers = []
    
    # FCC shells
    initializers.append(('fcc', generate_fcc_shells(size, spacing=1.0)))
    
    # HCP shells
    initializers.append(('hcp', generate_hcp_shells(size, spacing=1.0)))
    
    # Icosahedral
    initializers.append(('ico', generate_icosahedral(size, spacing=1.0)))
    
    # Random clusters with different seeds
    for seed in [0, 1, 2, 3]:
        initializers.append((f'rand{seed}', generate_random_cluster(size, spacing=1.0, seed=seed)))
    
    # Evaluate each initializer with local minimization
    best_energy = np.inf
    best_positions = None
    
    # First pass: local minimization on all initializers
    for name, init_pos in initializers:
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 30:
            break
        
        # Use relax from chem_clusters
        energy, positions = relax(init_pos, potential, rho)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Second pass: basin hopping on the best few
    if time.time() - start_time_global < TOTAL_TIME_BUDGET - 20:
        # Sort initializers by energy
        results = []
        for name, init_pos in initializers[:4]:  # Only top 4
            if time.time() - start_time_global > TOTAL_TIME_BUDGET - 25:
                break
            energy, positions = relax(init_pos, potential, rho)
            results.append((energy, name, positions))
        
        results.sort(key=lambda x: x[0])
        
        # Basin hopping on the best result
        if results:
            best_e, best_n, best_p = results[0]
            time_for_bh = min(40, TOTAL_TIME_BUDGET - 25 - (time.time() - start_time_global))
            if time_for_bh > 10:
                bh_pos, bh_energy = basin_hopping(
                    best_p, rho, 
                    n_iterations=30, 
                    step_size=0.4,
                    time_limit=time_for_bh
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
    
    # Final relaxation
    if best_positions is not None:
        energy, best_positions = relax(best_positions, potential, rho)
        best_energy = energy
    
    return best_positions
# EVOLVE-BLOCK-END
