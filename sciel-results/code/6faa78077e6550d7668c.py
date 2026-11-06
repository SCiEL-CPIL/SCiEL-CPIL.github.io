"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax

def _generate_fcc(n, noise=0.1):
    """Generate n points from an FCC lattice, centered at origin."""
    # FCC lattice vectors
    # Generate a large enough grid
    # FCC points are at (i, j, k) where i+j+k is even, scaled by a/2? 
    # Standard FCC primitive vectors: 
    # a1 = (a/2, a/2, 0), a2 = (a/2, 0, a/2), a3 = (0, a/2, a/2)
    # Or simpler: all (i, j, k) with integer coords where i+j+k is even.
    # Let's use the integer grid interpretation with spacing 1.0 (will scale later if needed, but Dzugutov units are fixed).
    # The equilibrium distance is ~1.1. Let's generate with spacing 1.1.
    spacing = 1.1
    
    points = []
    # Estimate radius needed for n atoms in FCC
    # Density of FCC is 4 atoms per (a^3) unit cell volume? No, 4 atoms per cubic unit cell of side a.
    # Volume per atom = a^3 / 4.
    # R^3 * 4/3 * pi * density = n => R = (3n / (4 pi density))^(1/3)
    # density = 4 / a^3 = 4 / (1.1^3) ~ 3.0
    # R ~ (3*60 / (4*3.14*3))^(1/3) ~ (180/37)^(1/3) ~ 4.8^(1/3) ~ 1.7? 
    # Wait, let's just generate a big box and filter.
    
    max_coord = 10
    grid = np.mgrid[-max_coord:max_coord+1, -max_coord:max_coord+1, -max_coord:max_coord+1]
    # Flatten
    coords = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
    # Filter for FCC: i+j+k is even
    mask = (coords[:,0].astype(int) + coords[:,1].astype(int) + coords[:,2].astype(int)) % 2 == 0
    fcc_coords = coords[mask]
    
    # Sort by distance from origin to get the most central cluster
    dists = np.linalg.norm(fcc_coords, axis=1)
    sorted_indices = np.argsort(dists)
    selected = fcc_coords[sorted_indices][:n]
    
    # Center of mass correction
    selected = selected - np.mean(selected, axis=0)
    
    # Add noise
    rng = np.random.default_rng()
    selected = selected + rng.normal(0, noise, selected.shape)
    
    return selected

def _generate_bcc(n, noise=0.1):
    """Generate n points from a BCC lattice, centered at origin."""
    spacing = 1.1
    max_coord = 10
    grid = np.mgrid[-max_coord:max_coord+1, -max_coord:max_coord+1, -max_coord:max_coord+1]
    coords = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
    # BCC: Corners (i,j,k) all even? No.
    # BCC points: (i,j,k) where i,j,k are integers, plus center (i+0.5, j+0.5, k+0.5).
    # Simpler: Generate integer grid. Keep (i,j,k) where i+j+k is even (FCC-like) OR just use integer grid for simple cubic?
    # Let's use Simple Cubic as a fallback, it's easier to generate.
    # Actually, let's stick to FCC and Random.
    return None

def _generate_random_dense(n, min_dist=0.9, radius_factor=0.8):
    """Generate n points via rejection sampling within a sphere."""
    rng = np.random.default_rng()
    # Estimate radius
    # Assume density ~ 1 atom per 1.3^3 volume?
    # Volume per atom ~ 1.1^3 * 1.5 (packing factor) ~ 2
    # R = (3n / (4*pi*0.5))^(1/3)
    radius = (3.0 * n / (4.0 * np.pi * 0.5)) ** (1.0/3.0) * radius_factor
    # Ensure radius is not too small
    radius = max(radius, 2.0)
    
    points = []
    attempts = 0
    max_attempts = n * 50
    while len(points) < n and attempts < max_attempts:
        # Sample in sphere
        # Use uniform in cube then reject outside sphere
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        
        # Check min distance
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
        
        points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fallback: just take whatever we have, or generate random without min dist check?
        # To ensure validity, we must have n points.
        # If rejection fails, we relax the constraint or just use random points in a box.
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        
    return np.array(points)

def construct(instance):
    """
    Finds the lowest-energy structure for a Dzugutov cluster of size n.
    Uses multiple initialization strategies (FCC lattice, Random Dense) and relaxes them.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    candidates = []
    
    # Strategy 1: FCC Lattice with noise
    # Try different noise levels
    for noise in [0.05, 0.1, 0.2]:
        pts = _generate_fcc(size, noise=noise)
        if pts is not None and len(pts) == size:
            candidates.append(pts)
            
    # Strategy 2: Random Dense Packing
    # Try different min distances
    for min_d in [0.8, 0.9, 1.0]:
        pts = _generate_random_dense(size, min_dist=min_d)
        if len(pts) == size:
            candidates.append(pts)
            
    # Strategy 3: Random Uniform in Sphere (loose)
    rng = np.random.default_rng(size)
    radius = 1.2 * size ** (1.0/3.0)
    pts = rng.uniform(-radius, radius, (size, 3))
    # Filter to sphere
    norms = np.linalg.norm(pts, axis=1)
    mask = norms <= radius
    # If too few, just take all (shouldn't happen with this radius)
    if np.sum(mask) >= size:
        pts = pts[mask][:size]
    else:
        pts = pts[:size]
    candidates.append(pts)
    
    best_energy = float('inf')
    best_positions = None
    
    # Relax all candidates
    # Limit the number of relaxations to stay within time budget
    # 300s / 6 instances = 50s per instance.
    # Relaxation is fast, so we can do all of them.
    for pos in candidates:
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback to seed behavior if all fail
        spacing = 1.1
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, best_positions = relax(np.array(points), potential, parameter)
        
    return best_positions

# EVOLVE-BLOCK-END
