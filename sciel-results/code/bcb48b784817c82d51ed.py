import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a proper Mackay icosahedral cluster.
    Shells:
    0: 1 atom (center)
    1: 12 atoms (vertices of icosahedron)
    2: 42 atoms (30 edge midpoints + 12 face centers)
    3: 92 atoms (120 vertices of dodecahedron? No, 120 is too many. 
       Actually Mackay 3rd shell has 92 atoms: 120 vertices of icosidodecahedron? 
       Let's check: 1, 13, 55, 147. 
       Shell 2 is 42 atoms. 1+12+42=55.
       Shell 3 is 92 atoms. 1+12+42+92=147.
       We need up to 60 atoms. So we use shells 0, 1, 2 and part of 3.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (radius 1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices
    if n > 1:
        points.extend([v * 1.0 for v in verts])
        
    # Shell 2: 42 atoms
    if n > 13:
        # Find edges (distance ~ 2.0 between normalized verts is 2.0 * 1/sqrt(1+phi^2)? No.
        # Distance between adjacent verts in unit icosahedron is 2.0 / sqrt(1+phi^2) * something?
        # Let's just compute distances and find the 30 shortest unique pairs.
        dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        # Sort distances
        dist_flat = dists.flatten()
        idx_flat = np.argsort(dist_flat)
        
        edges = []
        seen = set()
        for idx in idx_flat:
            i, j = divmod(idx, 12)
            if i == j: continue
            pair = (min(i,j), max(i,j))
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
                if len(edges) == 30:
                    break
        
        # Edge midpoints (30 atoms)
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid /= np.linalg.norm(mid)
            points.append(mid * 1.9) # Radius of shell 2 is sqrt(1+phi) ~ 1.9? 
            # Actually distance from center to edge midpoint of unit icosahedron is 1/sqrt(1+phi^2)?
            # Let's use a consistent scaling. 
            # If vertex is at 1.0, edge midpoint is at ~0.79? No, that's for a different scale.
            # Let's just project to a sphere of radius R2.
            # R2 should be such that it fits. 
            # In Mackay, the distance from center to vertex is a. 
            # Distance to edge midpoint is a * sqrt(1 - 1/(1+phi^2))? 
            # Let's just use 1.9 as a heuristic for now, or better, compute it.
            # Norm of (v1+v2)/2 where v1, v2 are unit vectors with angle theta.
            # cos(theta) for adjacent icosahedron verts is -1/sqrt(1+phi^2)? No.
            # Let's just use the normalized vector and multiply by a radius.
            # The radius of the 2nd shell in Mackay is sqrt(1+phi) * (1/sqrt(1+phi^2))? 
            # Let's just use 1.9.
            
        # Face centers (12 atoms)
        # We need the 20 faces. 
        # This is hard to generate programmatically without a mesh library.
        # However, we can approximate by adding 12 points on the sphere of radius R2 
        # that are roughly in the centers of the faces.
        # Or we can just add 12 random points on the sphere of radius 1.9? 
        # Better: use the 12 vertices of a dodecahedron? No.
        # Let's just add 12 points that are far from the edge midpoints.
        # Actually, the 12 face centers of an icosahedron form a dodecahedron.
        # The vertices of a dodecahedron are (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1).
        # These are the same as the icosahedron vertices but permuted? 
        # No, the icosahedron vertices are (0, ±1, ±phi) etc.
        # The dodecahedron vertices are (±1, ±1, ±1) and (0, ±1/phi, ±phi) etc.
        # Let's just generate 12 points on a sphere of radius 1.9 using a dodecahedral pattern.
        # Dodecahedron vertices (radius 1):
        dodeca_verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ])
        # Wait, a dodecahedron has 20 vertices, not 12. 
        # The 12 face centers of an icosahedron form a dodecahedron? 
        # Yes, the dual of an icosahedron is a dodecahedron. 
        # So the 12 face centers are the vertices of a dodecahedron? 
        # No, the 12 face centers are the 12 vertices of a dodecahedron? 
        # An icosahedron has 20 faces. A dodecahedron has 12 faces. 
        # So the 20 face centers of an icosahedron form a dodecahedron? No.
        # The 12 vertices of an icosahedron form a dodecahedron? No.
        # The dual of an icosahedron (20 faces, 12 vertices) is a dodecahedron (12 faces, 20 vertices).
        # So the 12 vertices of the icosahedron correspond to the 12 faces of the dodecahedron.
        # And the 20 faces of the icosahedron correspond to the 20 vertices of the dodecahedron.
        # So we need 12 points corresponding to the 12 faces of the dodecahedron? 
        # No, we need 12 points corresponding to the 12 vertices of the dual dodecahedron? 
        # The 12 face centers of the icosahedron are the 12 vertices of the dual dodecahedron? 
        # No, the dual of a polyhedron has vertices at the face centers of the original.
        # So the 20 face centers of the icosahedron are the 20 vertices of the dodecahedron.
        # But we only need 12 atoms in the 2nd shell besides the 30 edge midpoints. 
        # Total 42. 30 + 12 = 42. 
        # So we need 12 points. 
        # These 12 points are the 12 vertices of the icosahedron? No, those are in shell 1.
        # They must be the 12 face centers of the icosahedron? But there are 20 faces.
        # Wait, the Mackay icosahedron 2nd shell has 42 atoms. 
        # 30 are on the edges (midpoints). 
        # 12 are on the faces? But there are 20 faces. 
        # Ah, the 12 atoms are at the centers of the 12 faces of the dual dodecahedron? 
        # No, let's just look up the Mackay icosahedron structure.
        # Shell 2: 42 atoms. 30 are edge centers, 12 are face centers? 
        # Actually, 12 of the face centers are equivalent to the 12 vertices of the icosahedron? No.
        # Let's just add 12 random points on the sphere of radius 1.9 to complete the shell.
        # It's not perfect, but it's a good starting point.
        for _ in range(12):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
            points.append(p * 1.9)
            
    # Shell 3: 92 atoms
    if n > 55:
        needed = n - len(points)
        for _ in range(needed):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
            points.append(p * 2.4)
            
    return np.array(points[:n])

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points)

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    # 3 Mackay Icosahedral
    for _ in range(3):
        seeds.append(_generate_mackay_icosahedron(n, rng))
    # 3 Random
    for _ in range(3):
        seeds.append(_generate_random(n, rng))
        
    best_energy = float('inf')
    best_positions = seeds[0]
    
    # Phase 1: Relax all seeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # Phase 2: Basin Hopping on the best structure
    # If time permits
    current_energy = best_energy
    current_positions = best_positions
    
    # Define perturbation
    def perturb(positions, step_size):
        # Add Gaussian noise
        noise = rng.normal(0, step_size, positions.shape)
        new_positions = positions + noise
        return new_positions
        
    step_size = 0.5
    max_iters = 50
    
    for i in range(max_iters):
        if _get_time_remaining() < 5.0:
            break
            
        new_positions = perturb(current_positions, step_size)
        energy, relaxed = relax(new_positions, potential, parameter)
        
        if energy < current_energy:
            current_energy = energy
            current_positions = relaxed
            step_size *= 0.95
        else:
            step_size *= 1.05
            if step_size > 2.0:
                step_size = 2.0
                
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions

    return best_positions
