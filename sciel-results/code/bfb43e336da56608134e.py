import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
from collections import deque

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
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms (12 vertices + 30 edge midpoints of icosahedron)
    Shell 3: 120 atoms (more complex, approximated)
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

def _generate_random_compact(n, seed):
    """
    Generate a random compact cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * n ** (1.0/3.0)
    
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.8:
                continue
        points.append(p)
        attempts += 1
        
    if len(points) < n:
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
    deadline = start_time + 280  # 20s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    # Maintain a population of the best structures
    # Each entry is (energy, positions)
    population = []
    max_population = 8
    
    def _add_to_population(energy, positions):
        nonlocal best_energy, best_positions
        population.append((energy, positions))
        population.sort(key=lambda x: x[0])
        # Keep only the best max_population entries
        if len(population) > max_population:
            population = population[:max_population]
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
        return energy < best_energy
    
    def try_relax(init_positions, label=""):
        try:
            energy, positions = relax(init_positions, potential, None)
            _add_to_population(energy, positions)
            return True
        except Exception:
            pass
        return False

    # 1. Initial diverse starts
    initial_structures = [
        _generate_mackay_icosahedron(size),
        _generate_distorted_icosahedron(size, seed=size),
        _generate_polytetrahedral(size),
        _generate_fcc_cluster(size),
        _generate_random_compact(size, seed=size),
        _generate_random_compact(size, seed=size + 100),
        _generate_random_compact(size, seed=size + 200),
        _generate_distorted_icosahedron(size, seed=size + 300)
    ]
    
    for i, init in enumerate(initial_structures):
        if time.time() > deadline:
            break
        try_relax(init, f"initial_{i}")

    if best_positions is None:
        return _generate_mackay_icosahedron(size)

    # 2. Multi-start Basin Hopping with adaptive scales
    rng = np.random.default_rng(size + 1234)
    
    steps = 0
    max_steps = 200  # Hard cap, time will usually limit this first
    
    # Adaptive scale: start large, decrease over time
    # Use a geometric decay
    initial_scale = 0.5
    final_scale = 0.05
    decay_rate = (final_scale / initial_scale) ** (1.0 / max_steps)
    
    while time.time() < deadline - 10 and steps < max_steps:
        # Adaptive scale
        scale = initial_scale * (decay_rate ** steps)
        
        # Choose a parent from the population (weighted toward better structures)
        # Use a simple strategy: pick randomly from the top half of the population
        pop_size = len(population)
        if pop_size > 1:
            # Weighted selection: better structures have higher probability
            weights = [1.0 / (i + 1) for i in range(pop_size)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0
            parent_idx = 0
            for idx in range(pop_size):
                cumsum += weights[idx]
                if r <= cumsum:
                    parent_idx = idx
                    break
        else:
            parent_idx = 0
        
        parent_positions = population[parent_idx][1]
        
        # Generate perturbed structure
        perturbed = parent_positions + rng.normal(0, scale, parent_positions.shape)
        
        # Relax
        try_relax(perturbed, f"bh_{steps}")
        
        steps += 1
    
    return best_positions
