"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global start time for budget enforcement
_START_TIME = time.time()
TIME_LIMIT = 280.0  # Leave 20s safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_icosahedron_core(scale=1.0):
    """
    Generates the 13-atom icosahedral core.
    Returns array of shape (13, 3).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale
    verts = verts * scale
    # Add center
    center = np.array([[0, 0, 0]])
    return np.vstack([center, verts])

def _generate_shell(n_atoms, radius, rng):
    """
    Generates n_atoms randomly distributed on a spherical shell of given radius.
    """
    # Random points on sphere
    z = rng.uniform(-1, 1, n_atoms)
    r = np.sqrt(1 - z**2)
    theta = rng.uniform(0, 2*np.pi, n_atoms)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, z)) * radius
    return points

def _generate_icosahedral_cluster(n, rng, scale=1.0):
    """
    Generates an icosahedral cluster of n atoms.
    13 atoms in the core, remaining on a larger shell.
    """
    core = _generate_icosahedron_core(scale)
    if n <= 13:
        # If n < 13, we just take the first n atoms (center + some verts)
        # This is a weak case, but n>=38 so it's fine.
        return core[:n]
    
    remaining = n - 13
    # Place remaining atoms on a shell. 
    # In icosahedral order, the next shell is often at ~1.7 * R1.
    shell_radius = scale * 1.7 
    shell_atoms = _generate_shell(remaining, shell_radius, rng)
    
    return np.vstack([core, shell_atoms])

def _generate_icosahedral_cluster_v2(n, rng, scale=1.0):
    """
    Variant: Place remaining atoms on a slightly larger shell to avoid overlap.
    """
    core = _generate_icosahedron_core(scale)
    if n <= 13:
        return core[:n]
    
    remaining = n - 13
    shell_radius = scale * 2.0
    shell_atoms = _generate_shell(remaining, shell_radius, rng)
    
    return np.vstack([core, shell_atoms])

def _generate_bcc_cluster(n, rng, scale=1.0):
    """
    Generates a BCC-like cluster.
    """
    # BCC lattice vectors
    # Generate points in a box, keep those within a sphere
    # BCC: (0,0,0) and (0.5, 0.5, 0.5) in unit cell
    points = []
    # Estimate radius needed for n atoms
    # Volume ~ 4/3 pi R^3. Density ~ 2 atoms / a^3.
    # a ~ (2/n)^(1/3) * R^? 
    # Let's just generate a large box and filter.
    max_radius = scale * (n ** (1/3) * 1.5)
    # Grid step
    step = scale * 0.8 
    # Generate grid
    x = np.arange(-max_radius, max_radius, step)
    y = np.arange(-max_radius, max_radius, step)
    z = np.arange(-max_radius, max_radius, step)
    
    # This might be slow if max_radius is large. 
    # Better: generate random points in a sphere and snap to BCC lattice?
    # Or just random compact.
    
    # Let's do a simple random compact for now as BCC is not ideal for Dzugutov.
    # Dzugutov favors icosahedral.
    return _generate_random_compact(n, rng, scale * 1.2)

def _generate_random_compact(n, rng, scale=1.0):
    """
    Generates a random compact cluster with minimum distance constraint.
    """
    radius = scale * n ** (1.0 / 3.0)
    min_dist = 0.8 * scale
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
                attempts = 0
            else:
                attempts += 1
                if attempts > 100:
                    # Relax min_dist slightly if stuck
                    min_dist *= 0.95
                    attempts = 0
    if len(points) < n:
        # Fill remaining randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _rotate(points, rng):
    """
    Applies a random rotation to the points.
    """
    # Generate random rotation matrix
    # Using QR decomposition of a random Gaussian matrix
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.diag(R)
    ph = np.sign(d)
    ph[2] *= ph[1]
    Q *= ph
    return points @ Q.T

def construct(instance):
    """
    Finds the lowest-energy structure for the given cluster size.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    # Base scale for Dzugutov
    # The potential has a minimum around r ~ 1.0-1.2.
    # Let's use scale = 1.0 as the base unit.
    base_scale = 1.0
    
    rng = np.random.default_rng(size)
    
    # Generate a list of candidate initial structures
    candidates = []
    
    # 1. Icosahedral core + shell at 1.7R
    candidates.append(_generate_icosahedral_cluster(size, rng, base_scale))
    
    # 2. Icosahedral core + shell at 2.0R
    candidates.append(_generate_icosahedral_cluster_v2(size, rng, base_scale))
    
    # 3. Rotated Icosahedral core + shell at 1.7R
    cand3 = _generate_icosahedral_cluster(size, rng, base_scale)
    candidates.append(_rotate(cand3, rng))
    
    # 4. Rotated Icosahedral core + shell at 2.0R
    cand4 = _generate_icosahedral_cluster_v2(size, rng, base_scale)
    candidates.append(_rotate(cand4, rng))
    
    # 5. Random compact
    candidates.append(_generate_random_compact(size, rng, base_scale))
    
    # 6. Another random compact with different seed
    rng2 = np.random.default_rng(size + 1)
    candidates.append(_generate_random_compact(size, rng2, base_scale))
    
    # 7. Perturbed icosahedron
    cand7 = _generate_icosahedral_cluster(size, rng, base_scale)
    cand7 += rng.normal(0, 0.1, cand7.shape)
    candidates.append(cand7)
    
    best_energy = np.inf
    best_positions = None
    
    # Try each candidate
    for i, cand in enumerate(candidates):
        if not _check_time():
            break
            
        # Optional: Check if any atoms are too close (< 0.5) and fix them?
        # The relax function might handle this, but let's be safe.
        # The task says "atoms must be at least 0.5 apart".
        # If our initialization violates this, relax might fail or be slow.
        # Let's check pairwise distances.
        # For N=60, 60*59/2 = 1770 pairs. Fast enough.
        if np.linalg.norm(cand - cand.T, axis=1).min() < 0.5:
             # We need to fix overlaps. 
             # Simplest: add small random jitter and hope relax fixes it, 
             # or reject this candidate.
             # Let's try jittering.
             cand = cand + rng.normal(0, 0.05, cand.shape)
             # Re-check
             if np.linalg.norm(cand - cand.T, axis=1).min() < 0.5:
                 continue # Skip this candidate if still overlapping
        
        try:
            energy, positions = relax(cand, potential, instance.get("rho", instance.get("kind")))
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
        # If we have time, we can try to perturb the best solution found so far
        # and relax it again.
        if best_positions is not None and _check_time():
             # Small perturbation
             perturbed = best_positions + rng.normal(0, 0.05, best_positions.shape)
             # Check overlap
             if np.linalg.norm(perturbed - perturbed.T, axis=1).min() >= 0.5:
                 try:
                     e2, p2 = relax(perturbed, potential, instance.get("rho", instance.get("kind")))
                     if e2 < best_energy:
                         best_energy = e2
                         best_positions = p2
                 except Exception:
                     pass
             
    if best_positions is None:
        # Fallback: return the first candidate if relaxation failed
        return candidates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
