"""Polytetrahedral Dzugutov cluster optimization with BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def _dzugutov_energy_and_grad(pos):
    """Compute Dzugutov energy and gradient for a position array of shape (n,3)."""
    n = len(pos)
    energy = 0.0
    grad = np.zeros_like(pos)
    
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = pos[i] - pos[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue
            r_hat = r_vec / r
            
            # First part: r < a
            if r < a:
                exp_c = np.exp(c / (r - a))
                # V = A * (r^-m - B) * exp(c/(r-a))
                r_neg_m = r ** (-m)
                v = A * (r_neg_m - B) * exp_c
                # dV/dr = A * (-m * r^(-m-1) * exp_c + (r^-m - B) * exp_c * (-c/(r-a)^2))
                dv_dr = A * (-m * r ** (-(m+1)) * exp_c + (r_neg_m - B) * exp_c * (-c / (r - a) ** 2))
            elif r < b:
                exp_d = np.exp(d / (r - b))
                # V = B * exp(d/(r-b))
                v = B * exp_d
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                dv_dr = B * exp_d * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += v
            # Force on i is -dV/dr * r_hat (since r_vec = pos[i]-pos[j])
            # Gradient contribution to pos[i]: dv_dr * r_hat
            # Gradient contribution to pos[j]: -dv_dr * r_hat
            grad[i] += dv_dr * r_hat
            grad[j] -= dv_dr * r_hat
    
    return energy, grad


def _refine(pos, max_time=5.0):
    """Refine positions using BFGS with the Dzugutov energy."""
    pos = np.array(pos, dtype=float)
    
    def obj(x):
        pos_reshaped = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(pos_reshaped)
        return e, g.flatten()
    
    try:
        res = minimize(obj, pos.flatten(), jac=True, method='BFGS',
                       options={'maxiter': 300, 'gtol': 1e-6})
        if res.success or res.fun < float('inf'):
            return res.x.reshape(-1, 3), res.fun
    except Exception:
        pass
    
    # Fallback: use the provided relax function
    try:
        energy, positions = relax(pos, 'dzugutov')
        return positions, energy
    except Exception:
        pass
    
    return pos, _dzugutov_energy_and_grad(pos)[0]


def _generate_icosahedron(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
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
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _generate_dodecahedron(radius=1.0):
    """Generate 20 vertices of a regular dodecahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    # Dodecahedron vertices
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                verts.append([sx, sy, sz])
    for sx in [1, -1]:
        for sz in [1, -1]:
            verts.append([0, sx/phi, sx*phi])
        for sy in [1, -1]:
            verts.append([sy/phi, 0, sy*phi])
            verts.append([sy*phi, sy/phi, 0])
    
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_polytetrahedral(n, rng):
    """Build a polytetrahedral-inspired structure of n atoms.
    
    Uses icosahedral/dodecahedral building blocks arranged in a compact cluster.
    """
    if n <= 13:
        # 13-atom icosahedron: center + 12 vertices
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        if n < 13:
            # Take first n points
            return core[:n]
        return core
    
    points = []
    
    # Start with an icosahedral core
    core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
    points.extend(core)
    
    remaining = n - len(points)
    
    if remaining <= 0:
        return np.array(points[:n])
    
    # Add atoms in shells using icosahedral/dodecahedral symmetry
    # Shell radii chosen to maintain good distances
    shell_radii = [1.6, 2.2, 2.8, 3.4, 4.0]
    shell_idx = 0
    
    for _ in range(remaining):
        if shell_idx >= len(shell_radii):
            shell_idx = len(shell_radii) - 1
        
        radius = shell_radii[shell_idx]
        
        # Try to place atom on sphere with rejection sampling
        placed = False
        for attempt in range(50):
            # Random direction
            u = rng.uniform(-1, 1)
            v = rng.uniform(-1, 1)
            theta = np.arccos(u)
            phi_angle = 2 * np.pi * v
            p = np.array([
                radius * np.sin(theta) * np.cos(phi_angle),
                radius * np.sin(theta) * np.sin(phi_angle),
                radius * np.cos(theta)
            ])
            
            # Check minimum distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            
            if valid:
                points.append(p)
                placed = True
                break
        
        if not placed:
            # Try next shell or just add with slight perturbation
            shell_idx += 1
            if shell_idx < len(shell_radii):
                radius = shell_radii[shell_idx]
                p = np.array([
                    radius * rng.uniform(-1, 1),
                    radius * rng.uniform(-1, 1),
                    radius * rng.uniform(-1, 1)
                ])
                points.append(p)
            else:
                # Just add a random point
                p = np.array([
                    rng.uniform(-4, 4),
                    rng.uniform(-4, 4),
                    rng.uniform(-4, 4)
                ])
                points.append(p)
    
    return np.array(points[:n])


def _build_compact_random(n, rng):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    # Estimate radius for dense packing
    # Optimal nearest-neighbor distance for Dzugutov is around 1.87-1.94
    # For n atoms in a sphere: R ~ (3n/(4*pi*rho))^(1/3) where rho ~ 1/(1.9^3)
    R = 0.9 * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0) * 2.0
    
    points = []
    max_attempts = 2000
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
    
    # If not enough points, fill with random (relax will fix)
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
    
    return np.array(points)


def _build_layered(n, rng):
    """Build a layered structure inspired by polytetrahedral stacking."""
    points = []
    
    # First layer: icosahedron core
    if n >= 13:
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        points.extend(core)
    
    remaining = n - len(points)
    
    # Add layers above and below
    layer_z = [1.5, -1.5, 2.5, -2.5, 3.5, -3.5]
    layer_idx = 0
    
    while remaining > 0 and layer_idx < len(layer_z):
        z = layer_z[layer_idx]
        layer_radius = 1.2 + 0.3 * layer_idx
        
        # Place atoms in a ring at this z-level
        n_in_layer = min(remaining, 8 + 2 * layer_idx)
        for i in range(n_in_layer):
            angle = 2 * np.pi * i / n_in_layer + rng.uniform(-0.1, 0.1)
            r = layer_radius * rng.uniform(0.8, 1.2)
            p = np.array([r * np.cos(angle), r * np.sin(angle), z])
            
            # Check distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
                remaining -= 1
                if remaining <= 0:
                    break
        
        layer_idx += 1
    
    # Fill remaining with random compact
    while len(points) < n:
        R = 0.9 * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0) * 2.0
        p = rng.uniform(-R, R, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Multiple diverse starting geometries
    generators = [
        ('polytetrahedral', _build_polytetrahedral),
        ('compact_random', _build_compact_random),
        ('layered', _build_layered),
    ]
    
    # Number of restarts per generator type
    n_restarts = 6
    
    for gen_name, gen_func in generators:
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            try:
                pos = gen_func(size, rng)
                
                # Ensure minimum distance constraint (soft)
                # Relax will handle small violations
                
                # Refine with BFGS
                refined_pos, energy = _refine(pos, max_time=3.0)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = refined_pos
                    
            except Exception:
                continue
    
    # If we still have time, do additional random restarts
    if time.time() < deadline - 20:
        for restart in range(8):
            if time.time() > deadline:
                break
            try:
                pos = _build_compact_random(size, rng)
                refined_pos, energy = _refine(pos, max_time=3.0)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = refined_pos
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: random compact
        pos = _build_compact_random(size, rng)
        try:
            refined_pos, energy = _refine(pos, max_time=5.0)
            return refined_pos
        except Exception:
            return pos
    
    return best_positions
# EVOLVE-BLOCK-END
