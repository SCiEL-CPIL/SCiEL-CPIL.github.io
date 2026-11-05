"""Polytetrahedral Dzugutov cluster optimization with BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

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
            
            if r < a:
                # V = A * (r^-m - B) * exp(c/(r-a))
                inv_r = 1.0 / r
                inv_r_m = inv_r ** m
                exp_term = np.exp(c / (r - a))
                V = A * (inv_r_m - B) * exp_term
                dV_dr = A * (-m * inv_r_m * inv_r - B * (-c) * exp_term / (r - a)**2)
                # Actually let's compute derivative more carefully
                # V(r) = A * (r^-m - B) * exp(c/(r-a))
                # dV/dr = A * (-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c)/(r-a)^2)
                dV_dr = A * (-m * inv_r_m * inv_r * exp_term + (inv_r_m - B) * exp_term * (-c) / (r - a)**2)
            elif r < b:
                # V = B * exp(d/(r-b))
                exp_term = np.exp(d / (r - b))
                V = B * exp_term
                dV_dr = B * exp_term * (-d) / (r - b)**2
            else:
                continue
            
            # grad_i += dV/dr * (pos[i] - pos[j]) / r
            # grad_j -= dV/dr * (pos[i] - pos[j]) / r
            factor = dV_dr / r
            grad[i] += factor * r_vec
            grad[j] -= factor * r_vec
            energy += V
    
    return energy, grad


def _refine(positions, deadline):
    """Refine positions using BFGS with Dzugutov energy gradient."""
    pos_flat = positions.flatten()
    
    def obj(x):
        p = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(obj, pos_flat, jac=True, method='L-BFGS-B', 
                         options={'maxiter': 200, 'gtol': 1e-6})
        if result.success or result.fun < float('inf'):
            return result.x.reshape(-1, 3), result.fun
    except Exception:
        pass
    
    # Fallback to relax
    try:
        energy, pos = relax(positions, 'dzugutov')
        return pos, energy
    except Exception:
        return positions, float('inf')


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
    # Cube vertices
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                verts.append([i, j, k])
    # Rectangle vertices (3 orientations)
    for i in [-1/phi, 1/phi]:
        for j in [-phi, phi]:
            verts.append([0, i, j])
            verts.append([i, j, 0])
            verts.append([j, 0, i])
    
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_polytetrahedral(n, rng, seed_radius=1.0):
    """Build a polytetrahedral structure with nested icosahedra."""
    # Start with central atom
    points = [np.zeros(3)]
    
    # Add icosahedral shells
    shell_radii = [seed_radius, seed_radius * 1.5, seed_radius * 2.0, seed_radius * 2.5]
    shell_idx = 0
    
    while len(points) < n and shell_idx < len(shell_radii):
        radius = shell_radii[shell_idx]
        ico = _generate_icosahedron(radius)
        
        for v in ico:
            if len(points) >= n:
                break
            # Check minimum distance
            valid = True
            for p in points:
                if np.linalg.norm(v - p) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(v)
        
        shell_idx += 1
    
    # If still not enough, add random points on a larger sphere
    if len(points) < n:
        outer_radius = shell_radii[-1] * 1.3
        while len(points) < n:
            u = rng.uniform(-1, 1)
            theta = np.arccos(u)
            phi_angle = 2 * np.pi * rng.uniform(0, 1)
            p = np.array([
                outer_radius * np.sin(theta) * np.cos(phi_angle),
                outer_radius * np.sin(theta) * np.sin(phi_angle),
                outer_radius * np.cos(theta)
            ])
            points.append(p)
    
    return np.array(points[:n])


def _build_dodecahedral(n, rng, seed_radius=1.0):
    """Build a structure based on dodecahedral motifs."""
    points = [np.zeros(3)]
    
    # Add dodecahedral shell
    radius = seed_radius
    dodec = _generate_dodecahedron(radius)
    
    for v in dodec:
        if len(points) >= n:
            break
        valid = True
        for p in points:
            if np.linalg.norm(v - p) < 0.5:
                valid = False
                break
        if valid:
            points.append(v)
    
    # Add icosahedral shell if needed
    if len(points) < n:
        radius2 = seed_radius * 1.6
        ico = _generate_icosahedron(radius2)
        for v in ico:
            if len(points) >= n:
                break
            valid = True
            for p in points:
                if np.linalg.norm(v - p) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(v)
    
    # Fill remaining with random points
    if len(points) < n:
        outer_radius = seed_radius * 2.2
        while len(points) < n:
            u = rng.uniform(-1, 1)
            theta = np.arccos(u)
            phi_angle = 2 * np.pi * rng.uniform(0, 1)
            p = np.array([
                outer_radius * np.sin(theta) * np.cos(phi_angle),
                outer_radius * np.sin(theta) * np.sin(phi_angle),
                outer_radius * np.cos(theta)
            ])
            points.append(p)
    
    return np.array(points[:n])


def _build_compact_random(n, rng):
    """Generate n random points in a compact sphere."""
    R = 0.8 * (n / (4/3 * np.pi)) ** (1/3) * 2.0
    points = []
    max_attempts = 500
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
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
    
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
    
    return np.array(points)


def _build_layered(n, rng):
    """Build a layered structure (FCC-like layers)."""
    points = []
    layer_spacing = 1.5
    atoms_per_layer = 8
    
    layer = 0
    while len(points) < n:
        # Generate a hexagonal layer
        for i in range(atoms_per_layer):
            if len(points) >= n:
                break
            angle = 2 * np.pi * i / atoms_per_layer
            r = 0.8 if i % 2 == 0 else 1.2
            p = np.array([
                r * np.cos(angle),
                r * np.sin(angle),
                layer * layer_spacing
            ])
            # Check distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
        
        layer += 1
        if layer > 10:
            break
    
    # Fill remaining
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    rng = np.random.default_rng(size)
    
    # Define structural generators
    generators = [
        lambda: _build_polytetrahedral(size, rng, 1.0),
        lambda: _build_polytetrahedral(size, rng, 0.8),
        lambda: _build_polytetrahedral(size, rng, 1.2),
        lambda: _build_dodecahedral(size, rng, 1.0),
        lambda: _build_dodecahedral(size, rng, 0.9),
        lambda: _build_compact_random(size, rng),
        lambda: _build_compact_random(size, rng),
        lambda: _build_layered(size, rng),
    ]
    
    for i, gen in enumerate(generators):
        if time.time() > deadline:
            break
        
        try:
            points = gen()
            
            # First try BFGS refinement
            refined_pos, energy = _refine(points, deadline)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos
            
            # Also try relax as a second refinement
            if time.time() < deadline - 15:
                try:
                    e2, pos2 = relax(refined_pos, 'dzugutov')
                    if e2 < best_energy:
                        best_energy = e2
                        best_positions = pos2
                except Exception:
                    pass
                    
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        points = _build_compact_random(size, rng)
        try:
            energy, positions = relax(points, 'dzugutov')
            return positions
        except:
            return points
    
    return best_positions
# EVOLVE-BLOCK-END
