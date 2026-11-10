"""
Strategy: Diverse crystalline initialization + L-BFGS-B relaxation + best-of-many.
Morse clusters have known crystalline-like ground states. We generate multiple
starting points from FCC, BCC, HCP, and shell structures, relax each with
L-BFGS-B, and return the lowest-energy valid configuration.
"""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            exp_term = np.exp(rho * (1.0 - r))
            pair_e = exp_term * (exp_term - 2.0)
            energy += pair_e
            # dE/dr = exp_term * (2*exp_term - 2) * (-rho)
            dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
            # dE/dx_i = dE/dr * (x_i - x_j)/r
            factor = dE_dr / r
            grad[i] += factor * r_vec
            grad[j] -= factor * r_vec
    return energy, grad


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    points = []
    # BCC lattice points
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Sort by distance from center and take closest n
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    points = []
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
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
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-3, 4):
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
    # Use Fibonacci sphere for initial distribution, then relax
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * spacing * n ** (1.0/3.0) * 0.5, 
                       y * spacing * n ** (1.0/3.0) * 0.5, 
                       z * spacing * n ** (1.0/3.0) * 0.5])
    return np.array(points, dtype=float)


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points]) if points else np.inf
            if min_dist >= 0.7 * spacing:
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
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.fun, result.x.reshape(-1, 3)


def construct(instance):
    """Construct lowest-energy Morse cluster using diverse crystalline initializations."""
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
    ]
    
    best_energy = np.inf
    best_positions = None
    
    for gen_idx, gen_func in enumerate(generators):
        if time.time() > deadline:
            break
        
        try:
            init_pos = gen_func()
            if len(init_pos) < size:
                continue
            
            # Relax with L-BFGS-B
            energy, relaxed_pos = _relax_lbfgs(init_pos, rho, maxiter=150)
            
            # Check validity: minimum distance constraint
            valid = True
            for i in range(len(relaxed_pos)):
                for j in range(i + 1, len(relaxed_pos)):
                    if np.linalg.norm(relaxed_pos[i] - relaxed_pos[j]) < 0.5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # If no valid result, fall back to the library's relax
    if best_positions is None:
        init_pos = _generate_random_compact(size, seed=42)
        energy, best_positions = relax(init_pos, potential, rho)
    
    return best_positions
