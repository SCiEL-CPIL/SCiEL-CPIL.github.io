"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def _dzugutov_energy_and_grad(pos):
    """Compute Dzugutov energy and gradient for a cluster."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Mask for valid pairs
    mask = dist < b
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Only compute for pairs within cutoff
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                v = A * (r ** (-m) - B) * np.exp(c / (r - a))
                dv_dr = A * (-m * r ** (-m - 1) * np.exp(c / (r - a)) + (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2))
            elif r < b:
                v = B * np.exp(d / (r - b))
                dv_dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += v
            # Gradient contribution
            rij = pos[i] - pos[j]
            factor = dv_dr / r
            grad[i] -= factor * rij
            grad[j] += factor * rij
    
    return energy, grad


def _generate_icosahedron(radius=1.0):
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts


def _generate_tetrahedron(radius=1.0):
    """Generate vertices of a regular tetrahedron."""
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts


def _generate_random_cluster(n, spacing=1.1, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points, add them at random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_polytetrahedral(n, spacing=1.1):
    """Generate a polytetrahedral-like structure."""
    # Start with an icosahedron core
    ico = _generate_icosahedron(radius=spacing * 0.9)
    points = [ico[0]]  # Central atom
    
    # Add icosahedron vertices
    for v in ico[1:]:
        if len(points) < n:
            points.append(v)
    
    # Add more atoms in tetrahedral arrangements
    tet = _generate_tetrahedron(radius=spacing * 0.8)
    rng = np.random.default_rng(42)
    
    while len(points) < n:
        # Pick a random existing atom and add tetrahedral neighbors
        base_idx = rng.integers(0, len(points))
        base = points[base_idx]
        for v in tet:
            if len(points) < n:
                candidate = base + v
                # Check minimum distance
                if all(np.linalg.norm(candidate - q) >= 0.5 * spacing for q in points):
                    points.append(candidate)
    
    return np.array(points[:n])


def _local_minimize(pos, max_iter=200, tol=1e-8):
    """Simple gradient descent with adaptive step size."""
    pos = pos.copy()
    step_size = 0.01
    best_energy, _ = _dzugutov_energy_and_grad(pos)
    
    for it in range(max_iter):
        energy, grad = _dzugutov_energy_and_grad(pos)
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Adaptive step size
        new_pos = pos - step_size * grad
        new_energy, _ = _dzugutov_energy_and_grad(new_pos)
        
        if new_energy < energy:
            pos = new_pos
            step_size *= 1.1
            best_energy = new_energy
        else:
            step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return pos, best_energy


def construct(instance):
    """Find low-energy structures for Dzugutov clusters."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 280.0  # Safety margin
    
    # Generate diverse starting structures
    candidates = []
    
    # 1. Random compact clusters
    for seed in [size, size + 1, size + 2, size + 3, size + 4]:
        candidates.append(_generate_random_cluster(size, spacing=1.1, seed=seed))
    
    # 2. Polytetrahedral structures
    candidates.append(_generate_polytetrahedral(size, spacing=1.1))
    
    # 3. Icosahedral-like structure
    ico = _generate_icosahedron(radius=1.1 * size ** (1/3) * 0.6)
    if len(ico) >= size:
        candidates.append(ico[:size])
    else:
        # Pad with random points
        extra = _generate_random_cluster(size - len(ico), spacing=1.1, seed=size + 10)
        candidates.append(np.vstack([ico, extra]))
    
    best_energy = np.inf
    best_pos = None
    
    # Relax each candidate
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        try:
            # First try the provided relax function
            energy, relaxed_pos = relax(pos, potential, parameter)
            
            # Then do additional gradient descent
            refined_pos, refined_energy = _local_minimize(relaxed_pos, max_iter=100)
            
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_pos = refined_pos.copy()
            elif energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # If time remains, do more random restarts
    if best_pos is None or time.time() - start_time < time_limit - 10:
        rng = np.random.default_rng(12345)
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            pos = _generate_random_cluster(size, spacing=1.1, seed=rng.integers(0, 100000))
            try:
                energy, relaxed_pos = relax(pos, potential, parameter)
                refined_pos, refined_energy = _local_minimize(relaxed_pos, max_iter=50)
                
                if refined_energy < best_energy:
                    best_energy = refined_energy
                    best_pos = refined_pos.copy()
                elif energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
            except Exception:
                continue
    
    if best_pos is None:
        # Fallback: return a simple random cluster
        best_pos = _generate_random_cluster(size, spacing=1.1, seed=0)
    
    return best_pos
# EVOLVE-BLOCK-END
