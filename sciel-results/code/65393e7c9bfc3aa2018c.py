"""New program using structured seeds + diverse perturbations for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax


def _icosahedron_vertices():
    """Return vertices of a regular icosahedron centered at origin, edge length 1."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to edge length 1
    # Edge length is 2 for this construction
    verts /= 2.0
    return verts


def _make_icosahedral_cluster(n):
    """Create an icosahedral-like cluster of n atoms."""
    # Start with icosahedron core (12 atoms)
    core = _icosahedron_vertices()
    if n <= 12:
        # Just use a subset
        return core[:n].copy()
    
    # Add second shell: 30 atoms on faces/edges
    # For simplicity, place additional atoms on the surface
    # Use a simple approach: place atoms on a sphere with icosahedral symmetry
    # For now, just scale up and add random surface atoms
    radius = np.linalg.norm(core[0])
    # Add atoms on a slightly larger sphere
    rng = np.random.default_rng(42)
    additional = []
    for _ in range(n - 12):
        # Random direction
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        # Place at slightly larger radius
        r = radius * 1.5
        p = r * v
        # Check minimum distance
        all_pts = np.vstack([core, additional])
        if len(all_pts) > 0:
            dists = np.linalg.norm(all_pts - p, axis=1)
            if np.min(dists) < 0.8:
                continue
        additional.append(p)
    
    points = np.vstack([core, additional])
    return points


def _make_decahedral_cluster(n):
    """Create a decahedral-like cluster (5-fold twinned icosahedron)."""
    # Start with icosahedron
    core = _icosahedron_vertices()
    if n <= 12:
        return core[:n].copy()
    
    # Add atoms in a decahedral pattern
    # For simplicity, use icosahedron with extra layers
    radius = np.linalg.norm(core[0])
    rng = np.random.default_rng(123)
    additional = []
    for _ in range(n - 12):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = radius * (1.3 + 0.2 * rng.random())
        p = r * v
        all_pts = np.vstack([core, additional])
        if len(all_pts) > 0:
            dists = np.linalg.norm(all_pts - p, axis=1)
            if np.min(dists) < 0.8:
                continue
        additional.append(p)
    
    points = np.vstack([core, additional])
    return points


def _make_fcc_fragment(n):
    """Create an fcc-like fragment."""
    # Generate fcc lattice points
    points = []
    a = 1.0  # lattice constant, equilibrium distance
    # fcc: points at (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0) + translations
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # fcc basis
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i*a + dx*a, j*a + dy*a, k*a + dz*a])
                    if np.linalg.norm(p) <= 3.0:
                        points.append(p)
    
    points = np.array(points)
    # Keep only n closest to center
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add more
        rng = np.random.default_rng(99)
        while len(points) < n:
            p = rng.uniform(-3, 3, 3)
            if len(points) > 0:
                dists = np.linalg.norm(points - p, axis=1)
                if np.min(dists) < 0.8:
                    continue
            points = np.vstack([points, p])
    
    return points


def _make_random_compact(n, seed):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0/3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) > 0:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.8:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, add remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def construct(instance):
    """Find low-energy Morse cluster configuration using diverse seeds and local relaxation."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate multiple diverse starting configurations
    seeds = []
    
    # 1. Icosahedral
    try:
        seeds.append(_make_icosahedral_cluster(n))
    except:
        pass
    
    # 2. Decahedral
    try:
        seeds.append(_make_decahedral_cluster(n))
    except:
        pass
    
    # 3. FCC fragment
    try:
        seeds.append(_make_fcc_fragment(n))
    except:
        pass
    
    # 4-7. Random compact with different seeds
    for s in [1, 2, 3, 4]:
        try:
            seeds.append(_make_random_compact(n, seed=100 + s))
        except:
            pass
    
    # Relax each seed and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, seed_pts in enumerate(seeds):
        if len(seed_pts) != n:
            # Pad or trim
            if len(seed_pts) > n:
                seed_pts = seed_pts[:n]
            else:
                rng = np.random.default_rng(500 + i)
                while len(seed_pts) < n:
                    p = rng.uniform(-2, 2, 3)
                    seed_pts = np.vstack([seed_pts, p])
        
        try:
            energy, positions = relax(seed_pts, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # If all relaxations failed, return a simple valid configuration
    if best_positions is None:
        rng = np.random.default_rng(0)
        radius = n ** (1.0/3.0) * 0.7
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if len(points) > 0:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
