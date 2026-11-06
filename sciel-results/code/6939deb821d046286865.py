"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask out self-interactions (diagonal)
    np.fill_diagonal(dists, np.inf)
    e = np.exp(rho * (1.0 - dists))
    pair_energy = e * (e - 2.0)
    # Sum over i<j (avoid double counting)
    return 0.5 * np.sum(pair_energy)

def _morse_gradient(positions, rho):
    """Compute Morse gradient for a cluster."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    e = np.exp(rho * (1.0 - dists))
    # dE/dr = -2 * rho * e * (e - 1)
    # grad_r = (r_i - r_j) / |r_i - r_j|
    # grad_E_i = sum_j dE/dr * grad_r
    de_dr = -2.0 * rho * e * (e - 1.0)
    # de_dr[i,j] corresponds to force on i due to j
    # grad_E[i] = sum_j de_dr[i,j] * (pos[i] - pos[j]) / dists[i,j]
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad[i] += de_dr[i, j] * diff[i, j] / dists[i, j]
    return grad

def _generate_fcc(n):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(42)
    points = []
    # FCC lattice vectors
    a = 1.0
    lattice_vectors = [
        np.array([0, 0, 0]),
        np.array([0, a/np.sqrt(2), a/np.sqrt(2)]),
        np.array([a/np.sqrt(2), 0, a/np.sqrt(2)]),
        np.array([a/np.sqrt(2), a/np.sqrt(2), 0]),
        np.array([a, 0, 0]),
        np.array([0, a, 0]),
        np.array([0, 0, a]),
        np.array([a/2, a/2, 0]),
        np.array([a/2, 0, a/2]),
        np.array([0, a/2, a/2]),
    ]
    # Generate lattice points
    max_coord = int(np.ceil(n ** (1/3.0))) + 2
    lattice_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for lv in lattice_vectors:
                    p = np.array([i, j, k]) * a + lv
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    # Center the cluster
    centroid = np.mean(lattice_points, axis=0)
    lattice_points -= centroid
    
    # Select n closest points to origin
    dists = np.linalg.norm(lattice_points, axis=1)
    indices = np.argsort(dists)[:n]
    points = lattice_points[indices]
    return points

def _generate_icosahedron(n):
    """Generate an icosahedral cluster."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1)
    vertices = vertices / norms[:, np.newaxis]
    
    # Scale to appropriate radius
    radius = 0.5 * n ** (1/3.0)
    vertices = vertices * radius
    
    points = [vertices[0].copy()]
    rng = np.random.default_rng(123)
    
    # Add more points on concentric shells
    shell_radii = [radius * (i + 1) / (n ** (1/3.0)) for i in range(int(n ** (1/3.0)))]
    for r in shell_radii:
        while len(points) < n:
            # Random point on sphere of radius r
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * r
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n:
            break
    
    return np.array(points[:n])

def _generate_random_compact(n, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 0.75 * n ** (1/3.0)
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    # If we didn't get enough, add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def construct(instance):
    """Find low-energy arrangements of atoms bound by Morse potentials."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # FCC-like
    try:
        fcc = _generate_fcc(size)
        if len(fcc) == size:
            initial_configs.append(fcc)
    except:
        pass
    
    # Icosahedral
    try:
        ico = _generate_icosahedron(size)
        if len(ico) == size:
            initial_configs.append(ico)
    except:
        pass
    
    # Random compact with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        try:
            rand = _generate_random_compact(size, seed=seed)
            if len(rand) == size:
                initial_configs.append(rand)
        except:
            pass
    
    if not initial_configs:
        # Fallback: simple random
        rng = np.random.default_rng(0)
        radius = 0.75 * size ** (1/3.0)
        points = []
        for _ in range(size):
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        initial_configs.append(np.array(points))
    
    best_energy = np.inf
    best_positions = None
    
    # Relax each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(config, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    # If we have time, try L-BFGS-B on the best result
    if best_positions is not None and time.time() < deadline - 50:
        try:
            def objective_and_grad(x):
                x = x.reshape(-1, 3)
                e = _morse_energy(x, rho)
                g = _morse_gradient(x, rho).flatten()
                return e, g
            
            x0 = best_positions.flatten()
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 1000, 'gtol': 1e-8})
            if result.success or result.fun < best_energy:
                new_energy = result.fun
                new_positions = result.x.reshape(-1, 3)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = new_positions
        except:
            pass
    
    if best_positions is None:
        # Last resort: return the first initial config
        best_positions = initial_configs[0]
    
    return best_positions
# EVOLVE-BLOCK-END
