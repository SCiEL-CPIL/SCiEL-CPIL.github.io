"""Morse cluster global minima finder using structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 290.0  # Leave 10s margin for evaluator overhead
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
_start_time_global = time.time()

def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _morse_gradient(positions, rho):
    """Compute Morse energy gradient (vectorized)."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero
    dist[dist == 0] = 1e-12
    
    # e = exp(rho * (1 - r))
    e = np.exp(rho * (1.0 - dist))
    # dE/dr = -rho * e * (2e - 1)
    dEdr = -rho * e * (2.0 * e - 1.0)
    
    # Force on i from j is dEdr * (r_ij / r)
    # We need to sum over j for each i
    # F_i = sum_{j!=i} dEdr_ij * (r_i - r_j) / r_ij
    # Note: diff[i,j] = r_i - r_j
    
    # Create a mask for upper triangle to avoid double counting in gradient?
    # Actually, gradient is sum of forces.
    # Force on i from j: f_ij = dEdr_ij * (r_i - r_j) / r_ij
    # Force on j from i: f_ji = -f_ij
    
    # Vectorized approach:
    # dEdr is (n, n)
    # diff is (n, n, 3)
    # dist is (n, n)
    
    # We want grad[i] = sum_j dEdr[i,j] * diff[i,j] / dist[i,j]
    # Note: dEdr[i,j] is symmetric, diff[i,j] = -diff[j,i], dist[i,j]=dist[j,i]
    # So grad[i] = sum_{j!=i} dEdr[i,j] * diff[i,j] / dist[i,j]
    
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(dEdr, 0.0)
    np.fill_diagonal(dist, 1.0) # Just to avoid div by zero, though dEdr is 0
    
    # factor = dEdr / dist -> (n, n)
    factor = dEdr / dist
    # Broadcast factor to (n, n, 1)
    factor = factor[:, :, np.newaxis]
    # grad = sum over j of factor[i,j] * diff[i,j]
    grad = np.sum(factor * diff, axis=1)
    
    return grad

def _local_minimize(positions, rho, max_steps=2000, tol=1e-10):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def objective(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_steps, 'ftol': tol, 'gtol': 1e-8})
    best_pos = result.x.reshape(-1, 3)
    best_energy = result.fun
    
    # Verify with the provided relax function
    energy_relax, positions_relax = relax(positions.copy(), 'morse', rho)
    if energy_relax < best_energy:
        best_energy = energy_relax
        best_pos = positions_relax
    
    return best_energy, best_pos

def _generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    # Start with origin
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Generate FCC lattice points in shells
    max_coord = int(np.ceil(n ** (1/3.0)) * 1.5) + 3
    fcc_points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: points where i+j+k is even, scaled by spacing/2
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (spacing / 2.0)
                    fcc_points.append(p)
    
    # Sort by distance from origin
    fcc_points = np.array(fcc_points)
    distances = np.linalg.norm(fcc_points, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Select n closest points
    selected = fcc_points[sorted_indices[:n]]
    return selected

def _generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    # HCP layers
    layer_spacing = spacing * np.sqrt(2.0 / 3.0)
    in_layer_spacing = spacing
    
    # Generate layers
    max_layer = int(np.ceil(n ** (1/3.0))) + 3
    for layer_idx in range(-max_layer, max_layer + 1):
        z = layer_idx * layer_spacing
        # Hexagonal grid in this layer
        max_r = int(np.ceil(n ** (1/3.0))) + 3
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal coordinates
                x = (i + 0.5 * (j % 2)) * in_layer_spacing
                y = j * in_layer_spacing * np.sqrt(3.0) / 2.0
                p = np.array([x, y, z])
                points.append(p)
    
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    selected = points[sorted_indices[:n]]
    return selected

def _generate_mackay_icosahedra(n, spacing=1.0):
    """Generate n atoms in Mackay icosahedral shells."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    
    # Mackay icosahedra shells:
    # Shell 1: 1 atom (center)
    # Shell 2: 12 atoms (icosahedron vertices)
    # Shell 3: 42 atoms
    # Shell 4: 92 atoms
    # Shell 5: 162 atoms
    # Total: 1, 13, 55, 147, 309...
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 2: 12 vertices
    if n > 1:
        for v in ico_verts:
            points.append(v * spacing)
    
    # Shell 3: 42 points (edges of icosahedron + face centers? No, Mackay is specific)
    # For simplicity and robustness, we can approximate higher shells by projecting
    # lattice points onto spheres or using known coordinates.
    # A simpler robust approach for larger n: use a dense icosahedral-like grid.
    
    # Let's generate a dense set of points on icosahedral shells.
    # For n > 13, we add points from the next shell.
    # Shell 3 coordinates (42 points):
    # 30 edge midpoints? No, Mackay 309 cluster has shells 1, 12, 42, 92, 162.
    # Let's just generate a large set of candidate points and pick the closest n.
    
    # Generate a large icosahedral lattice
    max_coord = int(np.ceil(n ** (1/3.0)) * 1.5) + 3
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Icosahedral lattice basis is complex. 
                # Alternative: Use the FCC lattice but rotate/scale to mimic icosahedral symmetry?
                # Or just use a high-density random sphere and relax?
                # Given time constraints, let's stick to the 12 vertices + center + random fill for larger n.
                pass
                
    # If n is larger than 13, we need more points.
    # We will generate a Fibonacci sphere or random points on a sphere of radius ~ n^(1/3)
    if n > 13:
        # Add points on a sphere of radius R
        R = spacing * (n / 13) ** (1/3.0) * 1.2
        num_extra = n - len(points)
        # Use golden spiral for uniform distribution
        rng = np.random.default_rng(123)
        for _ in range(num_extra):
            # Random point on sphere
            u = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            r = np.sqrt(1 - u**2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = u
            points.append(np.array([x, y, z]) * R)
            
    points = np.array(points[:n])
    return points

def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, use a fallback
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _basin_hopping(positions, rho, n_restarts=5, max_steps_per_restart=500, 
                   delta_scale=0.5, time_limit=None):
    """Basin hopping with multiple restarts."""
    best_energy = _morse_energy(positions, rho)
    best_positions = positions.copy()
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Generate perturbation
        delta = delta_scale * rng.normal(size=positions.shape)
        perturbed = positions + delta
        
        # Local minimize
        energy, pos = _local_minimize(perturbed, rho, max_steps=max_steps_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
            positions = pos.copy()
    
    return best_energy, best_positions

def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    elapsed_global = time.time() - _start_time_global
    remaining_time = max(1.0, INSTANCE_TIME_BUDGET - (elapsed_global % INSTANCE_TIME_BUDGET))
    
    # Generate multiple initial structures
    initial_structures = []
    
    # 1. FCC shells
    try:
        fcc = _generate_fcc_shells(size)
        initial_structures.append(fcc)
    except Exception:
        pass
    
    # 2. HCP shells
    try:
        hcp = _generate_hcp_shells(size)
        initial_structures.append(hcp)
    except Exception:
        pass
    
    # 3. Mackay Icosahedral
    try:
        ico = _generate_mackay_icosahedra(size)
        initial_structures.append(ico)
    except Exception:
        pass
    
    # 4. Random compact structures
    for seed in [0, 1, 2, 3, 4, 5]:
        try:
            rand = _generate_random_compact(size, seed=seed)
            initial_structures.append(rand)
        except Exception:
            pass
    
    # Ensure we have at least one structure
    if not initial_structures:
        initial_structures.append(_generate_random_compact(size, seed=0))
    
    # Local minimize all initial structures
    best_energy = np.inf
    best_positions = None
    
    start_time = time.time()
    for i, struct in enumerate(initial_structures):
        if (time.time() - start_time) > remaining_time * 0.6:
            break
        
        try:
            energy, pos = _local_minimize(struct, parameter, max_steps=500)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    # If we have time, do basin hopping from the best structure
    if best_positions is not None and (time.time() - start_time) < remaining_time * 0.85:
        try:
            bh_time = remaining_time * 0.15
            bh_energy, bh_pos = _basin_hopping(best_positions, parameter, 
                                               n_restarts=5, max_steps_per_restart=300,
                                               time_limit=bh_time)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Final check with relax
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
