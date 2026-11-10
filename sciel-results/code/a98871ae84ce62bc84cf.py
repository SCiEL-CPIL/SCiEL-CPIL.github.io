"""
Strategy: Deterministic Multi-Start L-BFGS-B with Diverse Crystalline and Random Initializations.
Instead of Basin Hopping (which does local relax + kick + local relax repeatedly, wasting time on redundant local relaxations),
we perform a single L-BFGS-B relaxation from each of many diverse initial configurations.
This is materially different: it trades the iterative global search of basin hopping for a broader set of one-shot local minimizations.
The diversity comes from FCC, BCC, icosahedral, and random initializations, each relaxed once.
This allows more independent starts within the time budget, increasing the chance of finding the global minimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)  # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2) = e^(2x) - 2e^x
    # dV/dr = (2e^(2x) - 2e^x) * dx/dr = 2e^x(e^x - 1) * (-rho)
    # dV/dri = -dV/dr * (ri - rj) / r
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # Vectorized gradient calculation
    # dV_dr shape (n, n)
    # r shape (n, n)
    # We need to avoid division by zero on diagonal. 
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    # factor[i,j] = dV_dr[i,j] * inv_r[i,j]
    factor = dV_dr * inv_r
    
    # sum_j factor[i,j] -> row_sum (n,)
    row_sum = np.sum(factor, axis=1)
    
    # sum_j factor[i,j] * positions[j] -> factor @ positions (n, 3)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    a = np.sqrt(2.0)
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_icosahedron(n, rng):
    """Generate n atoms in an icosahedral shell structure."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi]
    ], dtype=float)
    
    # Normalize to unit sphere
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Scale to appropriate size
    scale = 1.5 * n ** (1/3)
    vertices *= scale
    
    points = [np.zeros(3)]  # Center atom
    
    # Add vertices
    for v in vertices:
        points.append(v.copy())
    
    # If we need more atoms, add shells
    shell_radius = scale * 1.5
    while len(points) < n:
        # Add random points on a sphere shell
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        p = shell_radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        points.append(p)
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_random(n, rng):
    """Generate n atoms randomly in a compact region."""
    radius = 1.0 * n ** (1.0 / 3.0) * 1.0
    points = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    points = np.array(points)
    points += rng.normal(0, 0.01, points.shape)
    return points

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. FCC
    fcc = _generate_fcc(size, rng)
    if fcc is not None:
        configs.append(fcc)
    
    # 2. BCC
    bcc = _generate_bcc(size, rng)
    if bcc is not None:
        configs.append(bcc)
    
    # 3. Icosahedron
    ico = _generate_icosahedron(size, rng)
    if ico is not None:
        configs.append(ico)
    
    # 4. Random (multiple with different seeds)
    for i in range(5):
        rng_i = np.random.default_rng(42 + i)
        rand = _generate_random(size, rng_i)
        configs.append(rand)
    
    best_energy = float('inf')
    best_positions = None
    
    # Perform L-BFGS-B relaxation from each configuration
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        
        try:
            final_pos, final_energy = _local_minimize(pos, parameter)
            
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception:
            # Fallback to simple relax if L-BFGS-B fails
            try:
                energy, positions = relax(pos, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If no valid position found, return a valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
