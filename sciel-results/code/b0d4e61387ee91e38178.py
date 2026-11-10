"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def dzugutov_energy(positions):
    """Compute total Dzugutov potential energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                # V(r) = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A*(-m*r^(-m-1))*exp(c/(r-a)) + A*(r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                # V(r) = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                dV_dr = 0.0
            
            # grad_i = dV/dr * (rij/r), grad_j = -dV/dr * (rij/r)
            if r > 1e-10:
                factor = dV_dr / r
                grad[i] += factor * rij
                grad[j] -= factor * rij
    return grad


def generate_icosahedron(center, scale=1.0):
    """Generate 13-atom icosahedron (1 center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of regular icosahedron
    vertices = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms * scale
    # Add center
    center_pos = np.array([0.0, 0.0, 0.0])
    all_points = np.vstack([center_pos.reshape(1, 3), vertices])
    all_points += center
    return all_points


def generate_decahedron(center, scale=1.0):
    """Generate 13-atom decahedron-like cluster (approximation)."""
    # Use a modified icosahedron with slightly different geometry
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Perturb slightly to break icosahedral symmetry
    rng = np.random.default_rng(42)
    vertices += rng.normal(0, 0.05, vertices.shape)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms * scale
    center_pos = np.array([0.0, 0.0, 0.0])
    all_points = np.vstack([center_pos.reshape(1, 3), vertices])
    all_points += center
    return all_points


def rotate(points, R):
    """Apply rotation matrix R to all points."""
    return points @ R.T


def random_rotation(rng):
    """Generate a random rotation matrix."""
    # Use QR decomposition of random Gaussian matrix
    M = rng.normal(0, 1, (3, 3))
    Q, Rmat = np.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


def assemble_cluster(n, rng, time_budget=45.0):
    """
    Divide-and-conquer assembly:
    1. Generate several small relaxed sub-clusters (icosahedra, decahedra)
    2. Combine them by optimizing relative positions and orientations
    3. Relax the full cluster
    """
    start_time = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Start with a single icosahedron and add atoms layer by layer
    # Strategy 2: Combine two icosahedra
    # Strategy 3: Random compact cluster with guided placement
    
    strategies = []
    
    # Strategy A: Icosahedron core + shell atoms
    if n >= 13:
        strategies.append(('ico_core', 13))
    if n >= 26:
        strategies.append(('two_ico', 26))
    if n >= 39:
        strategies.append(('three_ico', 39))
    
    # Strategy B: Random compact with minimum distance
    strategies.append(('random_compact', n))
    
    # Strategy C: Decahedron-based
    if n >= 13:
        strategies.append(('deca_core', 13))
    
    for strat_name, core_size in strategies:
        if time.time() - start_time > time_budget:
            break
        
        for attempt in range(3):
            if time.time() - start_time > time_budget:
                break
            
            try:
                if strat_name == 'ico_core':
                    # Start with icosahedron, add remaining atoms in shell
                    ico = generate_icosahedron(np.zeros(3), scale=1.1)
                    remaining = n - 13
                    if remaining <= 0:
                        positions = ico[:n]
                    else:
                        # Add atoms in a shell around the icosahedron
                        positions = np.vstack([ico])
                        for _ in range(remaining):
                            # Place new atom at random direction from center, at radius ~2.2
                            direction = rng.normal(0, 1, 3)
                            direction /= np.linalg.norm(direction)
                            new_pos = direction * 2.2 + rng.normal(0, 0.1, 3)
                            # Check minimum distance
                            if all(np.linalg.norm(new_pos - p) >= 0.55 for p in positions):
                                positions = np.vstack([positions, new_pos.reshape(1, 3)])
                            else:
                                # Try again with different radius
                                for _ in range(20):
                                    new_pos = direction * (2.0 + rng.uniform(0, 0.5)) + rng.normal(0, 0.15, 3)
                                    if all(np.linalg.norm(new_pos - p) >= 0.55 for p in positions):
                                        positions = np.vstack([positions, new_pos.reshape(1, 3)])
                                        break
                        positions = positions[:n]
                        
                elif strat_name == 'two_ico':
                    # Two icosahedra separated by ~3.0
                    ico1 = generate_icosahedron(np.array([-1.5, 0, 0]), scale=1.0)
                    ico2 = generate_icosahedron(np.array([1.5, 0, 0]), scale=1.0)
                    positions = np.vstack([ico1, ico2])[:n]
                    
                elif strat_name == 'three_ico':
                    # Three icosahedra in triangular arrangement
                    R_sep = 2.5
                    centers = [
                        np.array([0, 0, 0]),
                        np.array([R_sep, 0, 0]),
                        np.array([R_sep/2, R_sep*np.sqrt(3)/2, 0])
                    ]
                    all_pts = []
                    for c in centers:
                        all_pts.append(generate_icosahedron(c, scale=0.95))
                    positions = np.vstack(all_pts)[:n]
                    
                elif strat_name == 'deca_core':
                    deca = generate_decahedron(np.zeros(3), scale=1.1)
                    remaining = n - 13
                    positions = np.vstack([deca])
                    for _ in range(remaining):
                        direction = rng.normal(0, 1, 3)
                        direction /= np.linalg.norm(direction)
                        new_pos = direction * 2.2 + rng.normal(0, 0.1, 3)
                        if all(np.linalg.norm(new_pos - p) >= 0.55 for p in positions):
                            positions = np.vstack([positions, new_pos.reshape(1, 3)])
                        else:
                            for _ in range(20):
                                new_pos = direction * (2.0 + rng.uniform(0, 0.5)) + rng.normal(0, 0.15, 3)
                                if all(np.linalg.norm(new_pos - p) >= 0.55 for p in positions):
                                    positions = np.vstack([positions, new_pos.reshape(1, 3)])
                                    break
                    positions = positions[:n]
                    
                elif strat_name == 'random_compact':
                    # Random compact cluster
                    radius = 1.1 * n ** (1.0 / 3.0) * 0.8
                    points = []
                    max_tries = 10000
                    tries = 0
                    while len(points) < n and tries < max_tries:
                        p = rng.uniform(-radius, radius, 3)
                        if np.linalg.norm(p) <= radius:
                            if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                                points.append(p)
                        tries += 1
                    if len(points) < n:
                        # Fill remaining with small perturbations
                        while len(points) < n:
                            base = points[rng.integers(len(points))]
                            p = base + rng.normal(0, 0.3, 3)
                            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                                points.append(p)
                    positions = np.array(points)
                else:
                    continue
                
                # Ensure minimum distance constraint
                # Relax locally
                energy, relaxed_positions = relax(positions, 'dzugutov', None)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_positions
                    
            except Exception:
                continue
    
    # If all strategies failed, fall back to simple random
    if best_positions is None:
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        points = []
        tries = 0
        while len(points) < n and tries < 50000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
            tries += 1
        while len(points) < n:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.3, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        positions = np.array(points)
        energy, best_positions = relax(positions, 'dzugutov', None)
        best_energy = energy
    
    return best_positions, best_energy


def construct(instance):
    """Divide-and-conquer assembly for Dzugutov clusters."""
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Time budget per instance: 300s total for 6 instances, so ~45s each
    # Leave some margin
    time_per_instance = 42.0
    
    best_positions, best_energy = assemble_cluster(size, rng, time_budget=time_per_instance)
    
    # Additional local refinement with multiple random restarts if time allows
    # Try a few more random perturbations and relax
    for _ in range(2):
        perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
        # Ensure min distance
        valid = True
        for i in range(len(perturbed)):
            for j in range(i+1, len(perturbed)):
                if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            energy, relaxed = relax(perturbed, 'dzugutov', None)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
    
    return best_positions
# EVOLVE-BLOCK-END
