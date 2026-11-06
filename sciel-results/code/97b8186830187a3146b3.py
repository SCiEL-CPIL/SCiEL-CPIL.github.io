# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Upper triangle only
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    
    energy = np.zeros_like(r)
    # Region 1: r < a
    mask1 = r < a
    r1 = r[mask1]
    if len(r1) > 0:
        energy[mask1] = A * (r1**(-m) - B) * np.exp(c / (r1 - a))
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    r2 = r[mask2]
    if len(r2) > 0:
        energy[mask2] = B * np.exp(d / (r2 - b))
    # r >= b: zero
    
    return float(np.sum(energy))

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-10:
                continue
            rij = positions[i] - positions[j]
            if r < a:
                # dV/dr for r < a
                # V = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A*(-m*r^(-m-1)*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2))
                exp_term = np.exp(c / (r - a))
                dVdr = A * (-m * r**(-m-1) * exp_term + (r**(-m) - B) * exp_term * (-c / (r - a)**2))
            elif r < b:
                # V = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
            else:
                dVdr = 0.0
            
            # Force on i is -dV/dr * rij/r, on j is +dV/dr * rij/r
            factor = dVdr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    
    return grad

def _relax_lbfgsb(positions, max_iter=500, tol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    def fun(x):
        return _dzugutov_energy(x.reshape(-1, 3))
    def jac(x):
        return _dzugutov_gradient(x.reshape(-1, 3)).flatten()
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
        return res.x.reshape(-1, 3), res.fun
    except:
        return positions, _dzugutov_energy(positions)

def _generate_icosahedron_shell(radius, n_atoms):
    """Generate n_atoms on an icosahedral shell using Fibonacci sphere."""
    if n_atoms <= 0:
        return np.array([])
    # Fibonacci sphere for even distribution
    indices = np.arange(n_atoms)
    phi_angle = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1 - 2 * (indices + 0.5) / n_atoms
    radius_per = np.sqrt(1 - y**2)
    theta = phi_angle * indices
    x = np.cos(theta) * radius_per
    z = np.sin(theta) * radius_per
    points = np.column_stack([x, y, z]) * radius
    return points

def _generate_mackay_icosahedron(n):
    """Generate a Mackay icosahedral cluster with n atoms."""
    # Mackay icosahedron: 1 + 12 + 42 + 92 = 147 for full
    # For smaller n, use partial shells
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    r1 = 1.05
    if n > 1:
        points.extend([v * r1 for v in verts[:min(12, n-1)]])
    
    remaining = n - len(points)
    if remaining > 0:
        # Shell 2: 42 atoms (12 edge midpoints + 30 face centers)
        # Use Fibonacci sphere at larger radius
        r2 = 1.85
        shell2 = _generate_icosahedron_shell(r2, remaining)
        points.extend([p for p in shell2])
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _generate_bcc_cluster(n):
    """Generate a BCC-like cluster structure."""
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    
    lattice_points = np.array(lattice_points)
    centers = np.linalg.norm(lattice_points, axis=1)
    sorted_idx = np.argsort(centers)
    points = lattice_points[sorted_idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _generate_fcc_cluster(n):
    """Generate an FCC-like cluster structure."""
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i, j + 0.5, k + 0.5]))
                lattice_points.append(np.array([i + 0.5, j, k + 0.5]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k]))
    
    lattice_points = np.array(lattice_points)
    centers = np.linalg.norm(lattice_points, axis=1)
    sorted_idx = np.argsort(centers)
    points = lattice_points[sorted_idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _generate_polytetrahedral(n, rng):
    """Generate a polytetrahedral structure with multiple icosahedral motifs."""
    # Start with a central icosahedron
    points = _generate_mackay_icosahedron(min(n, 13))
    
    remaining = n - len(points)
    if remaining <= 0:
        return points[:n]
    
    # Add additional icosahedral motifs around the central one
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Place additional icosahedral units at face/edge positions
    while len(points) < n:
        # Pick a random direction and place a small icosahedral unit
        dir_vec = rng.normal(size=3)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        dist = 2.5 + rng.uniform(0, 0.5)
        center = dir_vec * dist
        
        # Add up to 13 atoms in a small icosahedral arrangement around center
        for k, v in enumerate(verts):
            if len(points) >= n:
                break
            offset = v * 1.0
            points.append(center + offset)
        if len(points) >= n:
            break
        # Also add a central atom for the unit
        if len(points) < n:
            points.append(center)
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, sigma, rng):
    """Perturb positions with Gaussian noise."""
    return positions + rng.normal(0, sigma, positions.shape)

def construct(instance):
    """Find low-energy Dzugutov cluster structures using basin-hopping with L-BFGS-B."""
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    rng = np.random.default_rng(n * 7919)
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Mackay icosahedron
    try:
        ico = _generate_mackay_icosahedron(n)
        initial_structures.append(ico)
    except:
        pass
    
    # 2. BCC
    try:
        bcc = _generate_bcc_cluster(n)
        initial_structures.append(bcc)
    except:
        pass
    
    # 3. FCC
    try:
        fcc = _generate_fcc_cluster(n)
        initial_structures.append(fcc)
    except:
        pass
    
    # 4. Polytetrahedral
    try:
        pt = _generate_polytetrahedral(n, rng)
        initial_structures.append(pt)
    except:
        pass
    
    # 5. Additional random polytetrahedral
    for _ in range(3):
        try:
            pt = _generate_polytetrahedral(n, rng)
            initial_structures.append(pt)
        except:
            pass
    
    # 6. Random compact
    radius = 1.2 * n**(1.0/3.0) * 0.7
    for _ in range(3):
        try:
            points = []
            attempts = 0
            while len(points) < n and attempts < 10000:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                        points.append(p)
                attempts += 1
            if len(points) == n:
                initial_structures.append(np.array(points))
        except:
            pass
    
    # Relax all initial structures and find the best
    best_energy = np.inf
    best_positions = None
    
    for init_pos in initial_structures:
        if time.time() > deadline - 20:
            break
        try:
            relaxed, energy = _relax_lbfgsb(init_pos, max_iter=300)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except:
            pass
    
    if best_positions is None:
        # Fallback
        return initial_structures[0]
    
    # Basin-hopping: perturb the best and re-relax
    # Adaptive perturbation size
    base_sigma = 0.15
    max_bh_steps = 2000  # Will be limited by time
    
    for step in range(max_bh_steps):
        if time.time() > deadline - 10:
            break
        
        # Adaptive sigma: decrease over time
        progress = step / max_bh_steps
        sigma = base_sigma * (1.0 - 0.5 * progress)
        
        # Perturb the best-so-far
        perturbed = _perturb(best_positions, sigma, rng)
        
        # Relax
        try:
            relaxed, energy = _relax_lbfgsb(perturbed, max_iter=200)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except:
            pass
        
        # Occasionally try from a random initial structure
        if step % 20 == 10 and step < max_bh_steps // 2:
            if initial_structures:
                rand_init = initial_structures[rng.randint(len(initial_structures))]
                perturbed2 = _perturb(rand_init, 0.3, rng)
                try:
                    relaxed2, energy2 = _relax_lbfgsb(perturbed2, max_iter=150)
                    if energy2 < best_energy:
                        best_energy = energy2
                        best_positions = relaxed2.copy()
                except:
                    pass
    
    return best_positions
# EVOLVE-BLOCK-END
