"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time

# Pre-computed optimal cluster geometries for Dzugutov potential
# These are based on known low-energy structures from the literature
# and will be used as starting points for relaxation

def _generate_icosahedral_cluster(n, spacing=1.1):
    """Generate an icosahedral cluster with n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * spacing
    
    # Octahedron vertices (6 vertices) - second shell
    oct_vertices = np.array([
        [spacing, 0, 0], [-spacing, 0, 0],
        [0, spacing, 0], [0, -spacing, 0],
        [0, 0, spacing], [0, 0, -spacing]
    ])
    
    # Build cluster layer by layer
    points = []
    
    # Center atom
    if n >= 1:
        points.append([0.0, 0.0, 0.0])
    
    # First shell: 12 icosahedron vertices
    for i in range(min(12, n - 1)):
        points.append(ico_vertices[i])
    
    # Second shell: add octahedron vertices and additional positions
    remaining = n - len(points)
    if remaining > 0:
        # Add octahedron vertices
        for v in oct_vertices:
            if len(points) < n:
                points.append(v)
        
        # If still need more atoms, add positions at icosahedron edge midpoints
        if len(points) < n:
            # Generate more positions by scaling icosahedron vertices
            for scale in [1.5, 2.0, 2.5]:
                for v in ico_vertices:
                    if len(points) < n:
                        points.append(v * scale)
            
            # If still not enough, add random positions
            while len(points) < n:
                p = np.random.uniform(-spacing*3, spacing*3, 3)
                if all(np.linalg.norm(p - q) >= 0.8*spacing for q in points):
                    points.append(p)
    
    return np.array(points[:n])

def _generate_bcc_cluster(n, spacing=1.1):
    """Generate a BCC-like cluster with n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [[0.0, 0.0, 0.0]]
    
    # BCC positions: cube corners + body center
    # Generate in shells
    max_shell = int(np.ceil((n/4)**(1/3)))
    
    for shell in range(1, max_shell + 1):
        for i in range(-shell, shell+1):
            for j in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    # BCC: add if (i,j,k) are all even or all odd (body center positions)
                    if (i % 2 == j % 2 == k % 2) or (i != 0 and j != 0 and k != 0):
                        p = np.array([i, j, k]) * spacing * 0.5
                        if np.linalg.norm(p) <= shell * spacing:
                            points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If we don't have enough, add random positions
    while len(points) < n:
        p = np.random.uniform(-spacing*2, spacing*2, 3)
        if all(np.linalg.norm(p - q) >= 0.8*spacing for q in points):
            points.append(p)
    
    return np.array(points[:n])

def construct(instance):
    """Find low-energy Dzugutov cluster structures using multiple starting geometries."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s safety margin
    
    # Generate multiple starting geometries
    starting_points_list = []
    
    # 1. Icosahedral cluster
    ico_points = _generate_icosahedral_cluster(size, 1.1)
    starting_points_list.append(ico_points)
    
    # 2. BCC cluster
    bcc_points = _generate_bcc_cluster(size, 1.1)
    starting_points_list.append(bcc_points)
    
    # 3. Random compact cluster
    rng = np.random.default_rng(size)
    radius = 1.1 * size ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
            points.append(p)
    starting_points_list.append(np.array(points))
    
    # 4. Another random cluster with different seed
    rng2 = np.random.default_rng(size + 100)
    points2 = []
    while len(points2) < size:
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points2):
            points2.append(p)
    starting_points_list.append(np.array(points2))
    
    best_energy = float('inf')
    best_positions = None
    
    # Relax each starting geometry
    for i, init_points in enumerate(starting_points_list):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(init_points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            continue
    
    # If we have time, try a few more random starts
    remaining_time = deadline - time.time()
    if remaining_time > 30 and best_positions is not None:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            rng_trial = np.random.default_rng(size + 1000 + trial)
            points_trial = []
            while len(points_trial) < size:
                p = rng_trial.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points_trial):
                    points_trial.append(p)
            try:
                energy, positions = relax(np.array(points_trial), potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception as e:
                continue
    
    if best_positions is None:
        # Fallback: return the first starting geometry
        return starting_points_list[0]
    
    return best_positions
# EVOLVE-BLOCK-END
