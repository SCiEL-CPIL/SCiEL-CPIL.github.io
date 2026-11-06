"""
Dzugutov cluster global minima search.

Strategy:
1. Generate diverse initial structures using polytetrahedral / icosahedral motifs.
2. Use L-BFGS-B local minimization with the provided relax function.
3. Basin hopping with adaptive perturbation to escape local minima.
4. Deadline guard: track wall-clock time and return best result found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    r = r[mask]
    
    energy = 0.0
    # First term: r < a
    mask_a = r < a
    if np.any(mask_a):
        ra = r[mask_a]
        energy += np.sum(A * (ra**(-m) - B) * np.exp(c / (ra - a)))
    # Second term: r < b (but r >= a to avoid double counting, though the formula applies for r < b)
    # Actually the potential is defined piecewise:
    # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a
    # V(r) = B exp(d/(r-b)) for r < b
    # But note: for r < a, both terms could apply? No, it's piecewise.
    # Let me re-read: V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    # This means for r < a: both terms? Or is it that the first term applies for r < a and second for a <= r < b?
    # The notation [r < a] and [r < b] suggests indicator functions.
    # So for r < a: V = A(r^-m - B)exp(c/(r-a)) + B exp(d/(r-b))
    # For a <= r < b: V = B exp(d/(r-b))
    # For r >= b: V = 0
    
    # Let me recompute properly
    energy = 0.0
    # First term for r < a
    mask_a = r < a
    if np.any(mask_a):
        ra = r[mask_a]
        energy += np.sum(A * (ra**(-m) - B) * np.exp(c / (ra - a)))
    # Second term for r < b
    mask_b = r < b
    if np.any(mask_b):
        rb = r[mask_b]
        energy += np.sum(B * np.exp(d / (rb - b)))
    
    return energy

def _energy_and_grad(positions):
    """Energy and gradient via the provided function."""
    e, g = dzugutov_energy_gradient(positions)
    return e, g

def _local_minimize(positions, maxiter=500):
    """Local minimization using L-BFGS-B."""
    n = len(positions)
    def fun(x):
        pos = x.reshape(n, 3)
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        mask = ~np.eye(n, dtype=bool)
        r = r[mask]
        if np.min(r) < 0.5:
            return 1e10, np.zeros_like(x)
        e, g = _energy_and_grad(pos)
        return e, g.reshape(-1)
    
    res = minimize(fun, positions.reshape(-1), method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'ftol': 1e-10})
    pos = res.x.reshape(n, 3)
    # Ensure minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    r = r[mask]
    if np.min(r) < 0.5:
        # Fallback: use original positions if constraint violated
        return _energy_and_grad(positions)[0], positions
    return res.fun, pos

def _generate_icosahedral_shell(n, spacing=1.1):
    """Generate atoms on an icosahedral shell structure."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]  # Core atom
    
    # Shell 1: 12 atoms at radius r1
    r1 = spacing
    for v in verts:
        points.append(r1 * v)
    
    if n > 13:
        # Shell 2: 42 atoms - approximate with additional icosahedral symmetry points
        # Use edge midpoints and face centers
        # For simplicity, use a denser spherical distribution
        remaining = n - len(points)
        # Fibonacci sphere for remaining atoms
        golden = (1 + np.sqrt(5)) / 2
        for i in range(remaining):
            z = 1 - 2 * (i + 0.5) / remaining
            radius = np.sqrt(max(0, 1 - z**2))
            theta = 2 * np.pi * i / golden
            r2 = spacing * 1.8
            points.append(r2 * np.array([radius * np.cos(theta), radius * np.sin(theta), z]))
    
    return np.array(points[:n])

def _generate_random_compact(n, spacing=1.1, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _basin_hopping(positions, n_iterations=20, perturbation=0.5, deadline=None):
    """Basin hopping algorithm."""
    n = len(positions)
    best_pos = positions.copy()
    best_energy = _local_minimize(positions)[0]
    
    rng = np.random.default_rng()
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        noise = rng.normal(0, perturbation, size=(n, 3))
        perturbed = best_pos + noise
        
        # Local minimize
        energy, pos = _local_minimize(perturbed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
            # Reduce perturbation on success
            perturbation *= 0.95
        else:
            # Increase perturbation on failure
            perturbation *= 1.1
    
    return best_energy, best_pos

def construct(instance):
    """Construct a low-energy Dzugutov cluster."""
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Generate diverse initial structures
    initial_structures = []
    
    # Icosahedral-based
    ico = _generate_icosahedral_shell(size)
    initial_structures.append(ico)
    
    # Random compact with different seeds
    for seed in [size, size + 1, size + 2, size + 3]:
        initial_structures.append(_generate_random_compact(size, seed=seed))
    
    # More diverse random structures
    for seed in [100 + size, 200 + size, 300 + size]:
        initial_structures.append(_generate_random_compact(size, spacing=1.0, seed=seed))
    
    best_energy = float('inf')
    best_positions = None
    
    for idx, init_pos in enumerate(initial_structures):
        if time.time() > deadline:
            break
        
        # Local minimize
        energy, pos = _local_minimize(init_pos)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
        
        # Basin hopping from this structure
        remaining_time = deadline - time.time()
        if remaining_time > 30:  # Only if we have enough time
            bh_energy, bh_pos = _basin_hopping(pos, n_iterations=10, 
                                                perturbation=0.3, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
    
    # Final local minimize to ensure we're at a local minimum
    if best_positions is not None and time.time() < deadline - 5:
        final_energy, final_pos = _local_minimize(best_positions)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
