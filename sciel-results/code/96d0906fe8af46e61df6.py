import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _generate_mackay_icosahedron(n, r1=1.0, r2=1.8, r3=2.6):
    """
    Generate a cluster structure based on Mackay icosahedral shells.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    if n > 1:
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    # Shell 2: 42 atoms
    if n > 13:
        base_verts = _get_icosahedron_vertices(1.0)
        
        # Find edges of the icosahedron
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        # 42 points: 12 vertices + 30 edge midpoints
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    # Shell 3: approximate with another scaled icosahedron if needed
    if n > 55:
        v3 = _get_icosahedron_vertices(r3)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(r3 + 0.5)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_distorted_icosahedron(n, seed):
    """
    Generate a distorted icosahedral structure.
    """
    rng = np.random.default_rng(seed)
    base = _generate_mackay_icosahedron(n)
    # Add random noise to break symmetry
    noise_scale = 0.2
    distorted = base + rng.normal(0, noise_scale, base.shape)
    return distorted

def _generate_polytetrahedral(n, r1=1.0, r2=1.8):
    """
    Generate a polytetrahedral structure.
    Dzugutov potential favors polytetrahedral order.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Start with a central atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Add tetrahedral shells
    # A tetrahedron has 4 vertices
    # For polytetrahedral, we can use a close-packed arrangement
    
    # Approximate by using icosahedral shells but with different radii
    # that favor tetrahedral coordination
    
    if n > 1:
        # First shell: 12 atoms (icosahedral)
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        # Second shell: 42 atoms
        base_verts = _get_icosahedron_vertices(1.0)
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    if n > 55:
        v3 = _get_icosahedron_vertices(2.6)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(3.1)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_random_compact(n, seed, min_dist=0.8):
    """
    Generate a random compact cluster.
    """
    rng = np.random.default_rng(seed)
    # Estimate radius based on packing
    # Volume ~ n * (min_dist/2)^3 * 4/3 pi
    # R ~ (3n / 4pi * (min_dist/2)^3)^(1/3)
    # Roughly R ~ n^(1/3) * min_dist
    radius = 1.2 * n ** (1.0/3.0) + min_dist
    
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
        points.append(p)
        attempts += 1
        
    if len(points) < n:
        # Fallback: just add random points, might be bad but better than crash
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points[:n])

def _generate_fcc_cluster(n):
    """
    Generate an FCC-like cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    a = 1.0  # lattice constant
    
    # Generate FCC points in a cube
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 1
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    if len(points) >= n:
                        break
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # Center the cluster
    if points:
        points = np.array(points[:n])
        center = np.mean(points, axis=0)
        points = points - center
    
    return points

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # 10s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    # List of (energy, positions) for top K structures
    top_k = []
    K = 5
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(init_positions, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
            
            # Update top K
            top_k.append((energy, positions))
            top_k.sort(key=lambda x: x[0])
            if len(top_k) > K:
                top_k = top_k[:K]
            
            return True
        except Exception:
            pass
        return False

    # 1. Initial diverse starts
    # Generate more random compact structures to cover more of the configuration space
    initial_structures = [
        _generate_mackay_icosahedron(size),
        _generate_distorted_icosahedron(size, seed=size),
        _generate_polytetrahedral(size),
        _generate_fcc_cluster(size),
    ]
    
    # Add 10 random compact structures
    for i in range(10):
        initial_structures.append(_generate_random_compact(size, seed=size + i * 100))
    
    for i, init in enumerate(initial_structures):
        if time.time() > deadline:
            break
        try_relax(init, f"initial_{i}")

    if best_positions is None:
        return _generate_mackay_icosahedron(size)

    # 2. Multi-Start Basin Hopping Loop
    rng = np.random.default_rng(size + 1234)
    
    # Define perturbation scales
    # Mix of small (polish) and large (escape)
    scales = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    steps = 0
    max_steps = 200 # Hard cap, time will usually limit this first
    
    while time.time() < deadline - 5 and steps < max_steps:
        # Choose a scale
        scale = scales[rng.integers(len(scales))]
        
        # Choose a starting structure from top K
        # Bias towards the best, but allow exploration
        # Simple: pick random from top K
        idx = rng.integers(len(top_k))
        start_energy, start_positions = top_k[idx]
        
        # Generate perturbed structure
        perturbed = start_positions + rng.normal(0, scale, start_positions.shape)
        
        # Relax
        improved = try_relax(perturbed, f"bh_{steps}")
        
        steps += 1
        
    return best_positions
