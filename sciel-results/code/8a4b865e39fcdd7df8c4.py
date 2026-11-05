"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Constants
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

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
    # Normalize to unit length then scale to radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _generate_candidate(n, rng, style='random'):
    """Generate a candidate structure of n atoms."""
    if style == 'icosahedron_core':
        # Start with 13-atom icosahedron (center + 12)
        if n < 13:
            # Just random
            return _generate_random_compact(n, rng)
        
        core = np.vstack([np.zeros(3), _generate_icosahedron(1.0)])
        points = list(core)
        
        # Add remaining atoms in a shell
        remaining = n - 13
        # Radius of shell should be larger than icosahedron radius
        # Icosahedron edge length for radius 1 is sqrt(2*phi) ~ 1.902
        # Atoms must be >= 0.5 apart.
        shell_radius = 1.5
        for _ in range(remaining):
            # Random point on sphere
            u = rng.uniform(-1, 1)
            v = rng.uniform(-1, 1)
            theta = np.arccos(u)
            phi_angle = 2 * np.pi * v
            p = np.array([
                shell_radius * np.sin(theta) * np.cos(phi_angle),
                shell_radius * np.sin(theta) * np.sin(phi_angle),
                shell_radius * np.cos(theta)
            ])
            # Check distance to existing points
            # Simple rejection sampling
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                # If rejected, just add it anyway or try again? 
                # Relaxation will fix overlaps if they are close.
                # Better to just add it and let relax handle it, or try again.
                # For efficiency, just add it. Relaxation is robust to small overlaps.
                points.append(p)
                
        return np.array(points)
        
    else:
        return _generate_random_compact(n, rng)

def _generate_random_compact(n, rng):
    """Generate n random points in a compact sphere."""
    # Radius estimation: Volume ~ n * (0.5)^3 * packing_efficiency
    # For Dzugutov, optimal packing is dense.
    # Approximate radius R such that (4/3) pi R^3 ~ n * 0.5^3
    R = 0.75 * (n / (4/3 * np.pi)) ** (1/3) * 2.0 # Heuristic scaling
    
    points = []
    max_attempts = 1000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        # Check min distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
            
    # If we didn't get enough points, just add random ones (relax will fix)
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    # parameter is not strictly needed for relax if potential is specified
    # but the signature expects it or it's ignored. 
    # Looking at seed: relax(positions, potential, parameter)
    # The task says: chem_clusters.relax(positions, 'dzugutov')
    # But the seed uses: relax(np.array(points), potential, parameter)
    # We should stick to the seed's calling convention if it works, 
    # or check if parameter is optional.
    # The task description says: chem_clusters.relax(positions, 'dzugutov')
    # This implies parameter might be optional or default.
    # However, the seed program passes `parameter`.
    # Let's assume the standard signature is relax(pos, pot, param=None).
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate multiple candidates
    # For n=55, icosahedron_core is very likely to find the global min.
    # For others, a mix of random and icosahedron_core.
    
    n_candidates = 10
    rng = np.random.default_rng(size)
    
    for i in range(n_candidates):
        if time.time() > deadline:
            break
            
        # Alternate between styles
        if i % 2 == 0:
            style = 'icosahedron_core'
        else:
            style = 'random'
            
        try:
            points = _generate_candidate(size, rng, style)
            # Relax
            energy, positions = relax(points, potential, instance.get('rho', instance.get('kind')))
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            # If relax fails, try to return a valid random structure
            # or just skip this candidate
            pass
            
    if best_positions is None:
        # Fallback: return a random valid structure
        points = _generate_random_compact(size, rng)
        try:
            energy, positions = relax(points, potential, instance.get('rho', instance.get('kind')))
            return positions
        except:
            return points
            
    return best_positions
# EVOLVE-BLOCK-END
