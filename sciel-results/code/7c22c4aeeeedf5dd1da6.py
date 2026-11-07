"""
Strategy: Physics-informed templates (FCC, HCP, BCC, Icosahedral, Random) + Basin Hopping.
Sutton-Chen potentials for transition metals typically favor FCC or HCP local order.
We generate fragments of ideal FCC, HCP, BCC lattices, icosahedral shells, and random compact clusters.
We then perform basin-hopping: perturb the cluster, relax with L-BFGS-B, and keep the best.
This exploits the known structural preferences of the potential.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

# Deadline guard: Evaluator kills at 330s. We aim for 295s to be safe.
_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate a large enough FCC grid
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Lattice constant a. Nearest neighbor distance d = a/sqrt(2).
    # We want spacing ~ d. So a = spacing * sqrt(2).
    a = spacing * np.sqrt(2)
    
    # Determine grid size
    # Volume ~ n * (spacing^3). Side length L ~ n^(1/3) * spacing.
    # Number of grid points per side ~ L / a.
    L = n ** (1.0/3.0) * spacing * 1.5
    n_side = int(np.ceil(L / a)) + 2
    
    # Generate all FCC points in a cube
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    points.append(p)
    
    points = np.array(points)
    
    # Filter to a sphere of radius R
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        # Expand grid if not enough points
        n_side += 2
        points = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        points.append(p)
        points = np.array(points)
        R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    # Sort by distance from center to get a compact cluster
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    # Center
    points = points - np.mean(points, axis=0)
    
    return points

def _hcp_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    # HCP structure
    # Layer 1: Hexagonal packing
    # Layer 2: Hexagonal packing shifted
    # Ideal c/a ratio = sqrt(8/3) ≈ 1.633
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    
    points = []
    
    # Determine number of layers
    # Approximate volume per atom ~ a^3 * c / 2 (2 atoms per unit cell)
    # Total volume ~ n * a^3 * c / 2
    # Height H ~ n * c / 2
    n_layers = int(np.ceil(np.sqrt(n))) + 2
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        # Shift for alternating layers
        if layer % 2 == 0:
            shift = np.array([0, 0])
        else:
            # Shift to fit into holes
            shift = np.array([a/2, a*np.sqrt(3)/6])
            
        # Hexagonal grid in xy plane
        # Points: (i*a + j*a/2, j*a*sqrt(3)/2)
        n_side = int(np.ceil(np.sqrt(n))) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift[0]
                y = j * a * np.sqrt(3) / 2.0 + shift[1]
                points.append([x, y, z])
                
    points = np.array(points)
    
    # Filter to a sphere
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        # Expand grid if not enough points
        n_layers += 2
        points = []
        for layer in range(-n_layers, n_layers + 1):
            z = layer * c
            if layer % 2 == 0:
                shift = np.array([0, 0])
            else:
                shift = np.array([a/2, a*np.sqrt(3)/6])
            n_side = int(np.ceil(np.sqrt(n))) + 2
            for i in range(-n_side, n_side + 1):
                for j in range(-n_side, n_side + 1):
                    x = i * a + j * a * 0.5 + shift[0]
                    y = j * a * np.sqrt(3) / 2.0 + shift[1]
                    points.append([x, y, z])
        points = np.array(points)
        R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points = points - np.mean(points, axis=0)
    
    return points

def _bcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    points = []
    n_side = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                p1 = base
                p2 = base + np.array([0.5, 0.5, 0.5]) * a
                points.append(p1)
                points.append(p2)
    
    points = np.array(points)
    
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        # Expand grid if not enough points
        n_side += 2
        points = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    p1 = base
                    p2 = base + np.array([0.5, 0.5, 0.5]) * a
                    points.append(p1)
                    points.append(p2)
        points = np.array(points)
        R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
        
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points = points - np.mean(points, axis=0)
    
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Generate an icosahedral-like cluster (13-atom icosahedron + shells)."""
    if rng is None:
        rng = np.random.default_rng()
        
    phi = (1 + np.sqrt(5)) / 2
    ico_atoms = np.array([
        [0, 0, 0],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    ])
    
    # Normalize to unit icosahedron (distance from center to vertex is 1)
    norms = np.linalg.norm(ico_atoms, axis=1)
    norms[norms == 0] = 1
    ico_atoms = ico_atoms / norms[:, np.newaxis]
    
    # Scale to typical spacing
    ico_atoms *= spacing
    
    if n <= 13:
        # Take closest n to center
        dists = np.linalg.norm(ico_atoms, axis=1)
        idx = np.argsort(dists)[:n]
        return ico_atoms[idx].copy()
    
    # Add shells
    atoms = [ico_atoms.copy()]
    current_n = 13
    shell_radius = spacing * 1.4
    while current_n < n:
        # Add random points on a shell
        for _ in range(n - current_n):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z * z)
            p = shell_radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
            atoms.append(p)
            current_n += 1
    all_atoms = np.vstack(atoms)
    all_atoms -= all_atoms.mean(axis=0)
    return all_atoms

def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 100
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
            attempts += 1
            
    # If we couldn't fill, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.5):
    """Perturb positions with Gaussian noise."""
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Spacing based on potential
    # Sutton-Chen equilibrium distance is around 1.0-1.2 in reduced units?
    # The task says "reduced units (lattice constant 1)".
    # For SC, min energy is at r ~ 1.0.
    spacing = 1.0
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Initial templates
    templates = []
    
    # FCC
    try:
        fcc = _fcc_fragment(size, spacing, rng)
        if len(fcc) == size:
            templates.append(fcc)
    except:
        pass
        
    # HCP
    try:
        hcp = _hcp_fragment(size, spacing, rng)
        if len(hcp) == size:
            templates.append(hcp)
    except:
        pass
        
    # BCC
    try:
        bcc = _bcc_fragment(size, spacing, rng)
        if len(bcc) == size:
            templates.append(bcc)
    except:
        pass
        
    # Icosahedral
    try:
        ico = _icosahedral_shell(size, spacing, rng)
        if len(ico) == size:
            templates.append(ico)
    except:
        pass
        
    # Random
    rand = _random_compact(size, spacing, rng)
    templates.append(rand)
    
    # Relax each template
    for t in templates:
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(t, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
            
    # Basin Hopping
    # We will perturb the best known position, relax, and keep the best.
    # We can also perturb other templates.
    # Let's do a simple loop.
    
    max_iterations = 200 # Arbitrary, limited by time
    
    for i in range(max_iterations):
        if _time_left() < 15.0:
            break
            
        # Choose a base to perturb
        # 50% chance to perturb best, 50% chance to perturb a random template
        if rng.random() < 0.5 and best_positions is not None:
            base = best_positions
        else:
            base = templates[rng.integers(len(templates))]
            
        # Perturbation scale: start larger, decrease?
        # Simple: constant scale
        scale = 0.5
        # Maybe scale down slowly
        # scale = 0.5 * (1.0 - i / max_iterations) + 0.1
        
        new_positions = _perturb(base, rng, scale)
        
        try:
            energy, positions = relax(new_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
            
    # Final high-precision polish pass: tighter tolerance, more iterations
    if best_positions is not None and _time_left() > 5.0:
        try:
            # Use scipy minimize with tighter tolerances for final polish
            def _objective_and_grad(x):
                pos = x.reshape(-1, 3)
                e, g = sutton_chen_energy_gradient(pos, parameter)
                return e, g.flatten()
            
            x0 = best_positions.flatten()
            result = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxcor': 50}
            )
            
            if result.success or result.fun < best_energy:
                polished = result.x.reshape(-1, 3)
                # Validate: check for non-finite coordinates and minimum distance
                if np.all(np.isfinite(polished)):
                    # Check minimum distance
                    min_dist = np.inf
                    for i in range(len(polished)):
                        for j in range(i+1, len(polished)):
                            d = np.linalg.norm(polished[i] - polished[j])
                            if d < min_dist:
                                min_dist = d
                    if min_dist >= 0.35:
                        best_energy = result.fun
                        best_positions = polished
        except:
            pass
            
    if best_positions is None:
        # Fallback
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
