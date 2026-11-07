"""Sutton-Chen transition-metal cluster global minima - growth-based construction."""
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n):
    """Generate positions for a truncated FCC cluster with n atoms.
    
    Uses the standard construction: place atoms at FCC lattice points
    within a sphere of radius R = (n/4)^(1/3) * a, where a is lattice spacing.
    For transition metal clusters, the equilibrium nearest-neighbor distance
    is approximately 1.0 in reduced units.
    """
    # For FCC, packing fraction: number of atoms ~ (4/3) pi R^3 / (a^3/2 * 2) ... 
    # Actually for FCC: density = 4 atoms / (a^3) per unit cell, so n ≈ 4 * (4/3)πR^3 / a^3
    # => R ≈ (3n/(16π))^(1/3) * a
    # But let's just use a simple approach: generate FCC points and take the n closest to origin
    
    # Generate FCC lattice points in a cube around origin
    a = 1.0  # lattice constant
    max_coord = int(np.ceil((3 * n / (16 * np.pi)) ** (1/3) * a)) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for (di, dj, dk) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + di) * a
                    y = (j + dj) * a
                    z = (k + dk) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _mackay_positions(n):
    """Generate positions for a Mackay icosahedron-based cluster.
    
    The Mackay icosahedron has shells:
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices of icosahedron)
    Shell 2: 42 atoms
    Shell 3: 92 atoms
    ...
    
    We'll construct shells incrementally.
    """
    # Icosahedron vertices (unit radius)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # normalize to unit sphere
    
    positions = [np.zeros(3)]  # center atom
    
    # Shell 1: 12 icosahedron vertices
    shell1_radius = 1.0
    for v in ico_verts:
        positions.append(v * shell1_radius)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Shell 2: 42 atoms - these are at the midpoints of edges and face centers of the icosahedron
    # For simplicity, we'll generate points on a sphere and select those closest to FCC-like positions
    # Actually, let's use a simpler approach: generate many candidate points on shell 2 radius
    # and pick the best n-13
    
    # Shell 2 radius for Mackay icosahedron: sqrt(1 + 1/phi^2) ≈ 1.25
    shell2_radius = np.sqrt(1 + 1/phi**2)
    
    # Generate candidate points: edge midpoints and face centers
    # Icosahedron has 30 edges and 20 faces
    # Edge midpoints: 30 points
    # Face centers: 20 points
    # Total: 50 candidates, but some may be too close; we'll generate on sphere
    
    # Better approach: use a Fibonacci sphere for shell 2 candidates
    n_candidates = 200
    golden_angle = np.pi * (3 - np.sqrt(5))
    candidates = []
    for i in range(n_candidates):
        z = 1 - 2 * i / (n_candidates - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        candidates.append([r * np.cos(theta), r * np.sin(theta), z])
    candidates = np.array(candidates) * shell2_radius
    
    # For each candidate, check minimum distance to existing positions
    existing = np.array(positions)
    for cand in candidates:
        if len(positions) >= n:
            break
        dists = np.linalg.norm(existing - cand, axis=1)
        if np.min(dists) > 0.3:  # minimum separation
            positions.append(cand)
            existing = np.array(positions)
    
    if len(positions) < n:
        # Add more points
        for i in range(n_candidates * 3):
            if len(positions) >= n:
                break
            z = 1 - 2 * i / (n_candidates * 3 - 1)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i * 1.7  # different spacing
            cand = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            existing = np.array(positions)
            dists = np.linalg.norm(existing - cand, axis=1)
            if np.min(dists) > 0.3:
                positions.append(cand)
    
    return np.array(positions[:n])


def _growth_construct(instance, max_iters_per_step=50):
    """Sequentially grow cluster by adding atoms at low-energy interstitial sites."""
    n = instance["n"]
    kind = instance["kind"]
    
    # Start with a small cluster and grow
    # Use FCC-like initial positions as a guide
    
    # Generate candidate interstitial positions on a grid/sphere
    def get_interstitial_candidates(existing_positions, radius=2.5):
        """Generate candidate positions for new atom."""
        n_existing = len(existing_positions)
        if n_existing == 0:
            return np.array([[0.0, 0.0, 0.0]])
        
        # Generate candidates: 
        # 1. On a sphere of various radii
        # 2. Near existing atoms (but not too close)
        candidates = []
        
        # Sphere candidates
        for r in np.linspace(0.5, radius, 5):
            n_pts = 50
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_pts):
                z = 1 - 2 * i / (n_pts - 1)
                rr = np.sqrt(1 - z**2)
                theta = golden_angle * i
                candidates.append([rr * np.cos(theta) * r, rr * np.sin(theta) * r, z * r])
        
        # Also add points near existing atoms (slightly displaced)
        for p in existing_positions:
            for d in np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3],
                               [0.2, 0.2, 0], [0.2, 0, 0.2], [0, 0.2, 0.2],
                               [-0.3, 0, 0], [0, -0.3, 0], [0, 0, -0.3]]):
                candidates.append(p + d)
        
        candidates = np.array(candidates)
        
        # Filter: must be at least 0.35 from all existing atoms
        valid = []
        for cand in candidates:
            dists = np.linalg.norm(existing_positions - cand, axis=1)
            if np.min(dists) >= 0.35:
                valid.append(cand)
        
        return np.array(valid) if valid else np.array([[radius, 0, 0]])
    
    # Start with 2 atoms
    if n <= 2:
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        else:
            return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Relax the initial 2-atom cluster
    try:
        energy, positions = relax(positions, 'sutton_chen', kind)
    except:
        pass
    
    # Grow to n atoms
    for current_n in range(2, n):
        # Get interstitial candidates
        candidates = get_interstitial_candidates(positions)
        
        best_energy = np.inf
        best_pos = None
        
        # Evaluate a subset of candidates (limit for speed)
        max_candidates = min(len(candidates), 30)
        if len(candidates) > max_candidates:
            # Sample uniformly
            indices = np.random.choice(len(candidates), max_candidates, replace=False)
            candidates = candidates[indices]
        
        for cand in candidates:
            test_positions = np.vstack([positions, cand.reshape(1, 3)])
            # Quick energy evaluation without full relaxation
            try:
                e, _ = sutton_chen_energy_gradient(test_positions, kind)
                if e < best_energy and np.isfinite(e):
                    best_energy = e
                    best_pos = cand
            except:
                continue
        
        if best_pos is None:
            # Fallback: add at a random position
            best_pos = np.array([1.5, 0.5, 0.3])
        
        positions = np.vstack([positions, best_pos.reshape(1, 3)])
        
        # Periodically relax to escape local minima
        if (current_n + 1) % 5 == 0 or current_n + 1 == n:
            try:
                energy, positions = relax(positions, 'sutton_chen', kind)
            except:
                pass
    
    return positions


def _multi_start_relax(instance, n_starts=8, time_budget=40):
    """Multi-start local optimization with diverse initial configurations."""
    n = instance["n"]
    kind = instance["kind"]
    
    import time
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_positions(n)
        configs.append(fcc)
    except:
        pass
    
    # 2. Mackay icosahedron-based
    try:
        mackay = _mackay_positions(n)
        configs.append(mackay)
    except:
        pass
    
    # 3. Random compact clusters
    rng = np.random.default_rng(42)
    for i in range(n_starts - 2):
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < n and attempts < 1000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.7 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == n:
            configs.append(np.array(points))
    
    # 4. Growth-based
    try:
        growth = _growth_construct(instance)
        configs.append(growth)
    except:
        pass
    
    # Relax each configuration
    for config in configs:
        if time.time() - start_time > time_budget:
            break
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            continue
    
    if best_positions is None:
        # Fallback: return FCC positions
        best_positions = _fcc_positions(n)
    
    return best_positions


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster configuration."""
    n = instance["n"]
    kind = instance["kind"]
    
    # Use multi-start approach with diverse initial configurations
    positions = _multi_start_relax(instance, n_starts=8, time_budget=45)
    
    # Final verification
    if not np.all(np.isfinite(positions)):
        positions = _fcc_positions(n)
    
    # Check minimum distance
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        # Need to fix: spread out atoms
        positions = _fcc_positions(n)
    
    return positions
