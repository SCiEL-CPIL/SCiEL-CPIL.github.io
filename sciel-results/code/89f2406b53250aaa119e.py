# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy efficiently using upper triangle only."""
    n = len(positions)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_energy_gradient_combined(positions, rho):
    """Compute Morse energy and gradient in one pass for scipy."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    # Avoid division by zero
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    x = rho * (1.0 - r_safe)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    
    # dE/dr = rho * exp(x) * (2*exp(x) - 2)
    dE_dr = rho * ex * (2.0 * ex - 2.0)
    
    # Gradient: for each pair (i,j), grad_i += dE_dr * (x_i - x_j)/r
    grad = np.zeros_like(positions)
    np.add.at(grad, iu[0], dE_dr[:, np.newaxis] * diff / r_safe[:, np.newaxis])
    np.add.at(grad, iu[1], -dE_dr[:, np.newaxis] * diff / r_safe[:, np.newaxis])
    
    return energy, grad


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    points = []
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    points = []
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for layer in range(-range_val, range_val + 1):
        z = layer * c / 2.0
        for i in range(-range_val, range_val + 1):
            for j in range(-range_val, range_val + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                points.append(np.array([x, y, z]))
                # B layer offset
                points.append(np.array([x + a * np.sqrt(3.0) / 6.0, 
                                        y + a / 2.0, z + c / 6.0]))
    
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_shell(n, spacing=1.0):
    """Generate a spherical shell-like cluster using Fibonacci sphere."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = [np.zeros(3)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = spacing * (n ** (1/3.0) * 0.5)
    for i in range(1, n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append(np.array([x, y, z]) * radius)
    
    return np.array(points)


def _relax_l_bfgs(positions, rho, max_iter=500, gtol=1e-6):
    """Relax using L-BFGS-B with combined energy/gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_gradient_combined(pos, rho)
        return e, g.ravel()
    
    x0 = positions.ravel()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 30})
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    return final_energy, final_pos


def _polish(positions, rho, max_iter=1000, gtol=1e-8):
    """High-precision polish pass with tighter tolerance."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_gradient_combined(pos, rho)
        return e, g.ravel()
    
    x0 = positions.ravel()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 50})
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    return final_energy, final_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 300s limit
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    generators = [
        _generate_fcc,
        _generate_bcc,
        _generate_hcp,
        _generate_shell,
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Try each lattice generator with standard relaxation
    for gen_func in generators:
        if time.time() > deadline - 60:
            break
        
        try:
            positions = gen_func(n, spacing=1.0)
            energy, relaxed = _relax_l_bfgs(positions, rho, max_iter=400, gtol=1e-6)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Phase 2: Random restarts if time permits
    if time.time() < deadline - 90:
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0 / 3.0) * 0.75
        n_restarts = 3
        for _ in range(n_restarts):
            if time.time() > deadline - 60:
                break
            
            positions = []
            for _ in range(n):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.6 for q in positions
                ):
                    positions.append(p)
                else:
                    # Retry a few times
                    for _ in range(5):
                        p = rng.uniform(-radius, radius, 3)
                        if np.linalg.norm(p) <= radius and all(
                            np.linalg.norm(p - q) >= 0.6 for q in positions
                        ):
                            positions.append(p)
                            break
            if len(positions) < n:
                # Pad with random points
                while len(positions) < n:
                    positions.append(rng.uniform(-radius, radius, 3))
            
            positions = np.array(positions)
            try:
                energy, relaxed = _relax_l_bfgs(positions, rho, max_iter=300, gtol=1e-5)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    # Phase 3: Perturb best solution
    if best_positions is not None and time.time() < deadline - 60:
        rng = np.random.default_rng(n + 42)
        for _ in range(4):
            if time.time() > deadline - 45:
                break
            
            noise = rng.normal(0, 0.25, size=best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, relaxed = _relax_l_bfgs(perturbed, rho, max_iter=300, gtol=1e-6)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    # Phase 4: Final high-precision polish
    if best_positions is not None and time.time() < deadline - 20:
        try:
            energy, polished = _polish(best_positions, rho, max_iter=800, gtol=1e-8)
            if energy < best_energy:
                best_energy = energy
                best_positions = polished
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0 / 3.0) * 0.75
        positions = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        best_positions = np.array(positions)
    
    return best_positions
# EVOLVE-BLOCK-END
