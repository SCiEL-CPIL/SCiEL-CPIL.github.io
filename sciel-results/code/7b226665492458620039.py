"""
Strategy: Vectorized Morse energy/gradient + basin hopping with diverse crystalline initializations.
Combines the speed of vectorized energy/gradient computation with basin hopping to escape
local minima, starting from multiple diverse crystalline structures.
"""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute all pair differences: shape (n, n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # Distances: shape (n, n)
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]  # shape (n*(n-1)/2, 3)
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(positions)
    
    # Avoid division by zero
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    # dE/dr = exp_term * (2*exp_term - 2) * (-rho)
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    # Gradient: for each pair (i,j), grad[i] += factor * (pos[i] - pos[j]), grad[j] -= factor * (pos[i] - pos[j])
    grad = np.zeros_like(positions)
    # Get indices for upper triangle
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Sort by distance from center and take closest n
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    lattice = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                # FCC: corners + face centers
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Remove duplicates by rounding
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    lattice = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-4, 5):
                # A layer
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = k * c
                lattice.append([x, y, z])
                # B layer (offset)
                x2 = a * (i + 0.5 * ((j + 1) % 2)) + a / 2.0
                y2 = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z2 = (k + 0.5) * c
                lattice.append([x2, y2, z2])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster (icosahedral-like)."""
    points = []
    # Use Fibonacci sphere for initial distribution
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0/3.0) * 0.45
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    min_dist_required = 0.6 * spacing
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                # Vectorized distance check
                dists = np.sqrt(np.sum((np.array(points) - p) ** 2, axis=1))
                if np.min(dists) >= min_dist_required:
                    points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points, dtype=float)


def _relax_lbfgs(positions, rho, maxiter=200):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
    return result.fun, result.x.reshape(-1, 3)


def _check_validity(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart using vectorized computation."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    # Check upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.min(r[mask]) >= min_dist


def _basin_hop(positions, rho, n_iter=30, step_size=0.1, rng=None, deadline=None):
    """
    Basin hopping: alternate between random perturbation and local minimization.
    Returns the best (lowest energy) valid configuration found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy = np.inf
    best_pos = positions.copy()
    
    # First relax the initial position
    energy, relaxed = _relax_lbfgs(positions, rho, maxiter=200)
    if _check_validity(relaxed) and energy < best_energy:
        best_energy = energy
        best_pos = relaxed.copy()
    
    current_pos = relaxed.copy()
    current_energy = energy
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Random perturbation
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax the perturbed position
        new_energy, new_pos = _relax_lbfgs(perturbed, rho, maxiter=150)
        
        # Accept if lower energy (Metropolis-like, but simplified: always accept lower)
        if _check_validity(new_pos):
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
            elif new_energy < best_energy:
                pass  # Still track best
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Construct lowest-energy Morse cluster using basin hopping from diverse crystalline initializations."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse starting points
    generators = [
        lambda: _generate_bcc(size),
        lambda: _generate_fcc(size),
        lambda: _generate_hcp(size),
        lambda: _generate_shell(size),
        lambda: _generate_random_compact(size, seed=42),
        lambda: _generate_random_compact(size, seed=123),
        lambda: _generate_bcc(size, spacing=0.95),
        lambda: _generate_bcc(size, spacing=1.05),
        lambda: _generate_fcc(size, spacing=0.95),
    ]
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(42)
    
    # Allocate time budget: give each generator a fair share, but allow early exit
    # For n_iter in basin_hop, we need to be careful about time
    # Estimate: each L-BFGS-B call takes ~0.1-0.5s for n<=80
    # So basin_hop with 30 iterations could take 3-15s per generator
    # With 9 generators, that's 27-135s total, which is within budget
    
    n_basin_iter = 25  # Number of basin hopping iterations per generator
    
    for gen_idx, gen_func in enumerate(generators):
        if time.time() > deadline - 10.0:
            break
        
        try:
            init_pos = gen_func()
            if len(init_pos) < size:
                continue
            
            # Basin hopping from this initialization
            energy, relaxed_pos = _basin_hop(init_pos, rho, n_iter=n_basin_iter, 
                                             step_size=0.08, rng=rng, deadline=deadline)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # If no valid result, fall back to the library's relax
    if best_positions is None:
        init_pos = _generate_random_compact(size, seed=42)
        energy, best_positions = relax(init_pos, potential, rho)
    
    return best_positions
