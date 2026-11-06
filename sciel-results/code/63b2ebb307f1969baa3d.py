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

def _generate_mackay_icosahedron(n, r1=1.0, r2=1.87):
    """
    Generate a cluster structure based on Mackay icosahedral shells.
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms (12 vertices + 30 edge midpoints of icosahedron)
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
    
    # If n > 55, we need to add more. 
    # For Dzugutov, the 55-atom Mackay is a strong local min.
    # For 57, 60, we can add atoms on a third shell or distort.
    if n > 55:
        # Add atoms on a third shell (icosahedron vertices at larger radius)
        # The 55-atom cluster has 12 vertices on shell 2.
        # We can add atoms near the 12 faces of the icosahedron or extend the shell.
        # A common 60-atom structure is a 55-atom core + 5 atoms, or a distorted 55.
        # Let's try adding atoms at the centers of the 20 faces of the icosahedron, scaled out.
        # Or simply add them at random positions on a sphere of radius ~2.6.
        
        # Face centers of icosahedron
        # The icosahedron has 20 faces.
        # We can approximate face centers by taking centroids of 3 adjacent vertices.
        # But simpler: just add atoms at positions that are far from existing ones.
        
        # Let's use a simple heuristic: add atoms at positions on a sphere of radius r3=2.6
        # that are as far as possible from existing atoms.
        
        # For now, just append random points on a sphere of radius 2.6
        # This is a rough seed. The relaxation will fix it.
        rng = np.random.default_rng(42)
        current_points = np.array(points)
        
        # Calculate current max radius
        max_r = np.max(np.linalg.norm(current_points, axis=1))
        
        for _ in range(n - len(points)):
            # Try to find a position
            added = False
            for _attempt in range(100):
                # Random direction
                u, v = rng.uniform(0, 1), rng.uniform(0, 1)
                theta = 2 * np.pi * u
                phi_angle = np.arccos(2 * v - 1)
                dir_vec = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ])
                # Radius slightly larger than current max
                r_new = max_r + rng.uniform(0.5, 1.0)
                p_new = dir_vec * r_new
                
                # Check distance to all existing points
                dists = np.linalg.norm(current_points - p_new, axis=1)
                if np.min(dists) > 1.5: # Ensure not too close
                    points.append(p_new)
                    current_points = np.array(points)
                    max_r = np.max(np.linalg.norm(current_points, axis=1))
                    added = True
                    break
            if not added:
                # Just add it, relaxation will sort it out
                points.append(current_points[0] + rng.normal(0, 1, 3))
    
    return np.array(points[:n])

def _generate_growing_icosahedron(n, seed=0):
    """
    Generate a cluster by growing an icosahedron.
    Start with 1 atom, add 12, then add more atoms one by one
    at positions that are roughly 1.87 away from existing atoms.
    """
    rng = np.random.default_rng(seed)
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Add first 12 atoms in icosahedral config
    ico12 = _get_icosahedron_vertices(1.0)
    for v in ico12:
        if len(points) < n:
            points.append(v)
    
    # Add remaining atoms
    # We want to add atoms that are roughly 1.87 away from some existing atoms
    # and as far as possible from others.
    
    current_points = np.array(points)
    
    for _ in range(n - len(points)):
        best_p = None
        best_score = -np.inf
        
        # Try several random candidate positions
        for _ in range(200):
            # Pick a random existing atom
            idx = rng.integers(0, len(points))
            ref_atom = points[idx]
            
            # Random direction
            u, v = rng.uniform(0, 1), rng.uniform(0, 1)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            dir_vec = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            
            # Distance ~1.87
            r = 1.87 + rng.normal(0, 0.1)
            p_new = ref_atom + dir_vec * r
            
            # Check min distance to all other atoms
            dists = np.linalg.norm(np.array(points) - p_new, axis=1)
            min_d = np.min(dists)
            
            if min_d < 1.2: # Too close
                continue
            
            # Score: prefer positions that are far from all atoms
            score = min_d
            if score > best_score:
                best_score = score
                best_p = p_new
        
        if best_p is not None:
            points.append(best_p)
        else:
            # Fallback: add random point
            points.append(np.array([rng.normal(0, 2), rng.normal(0, 2), rng.normal(0, 2)]))
            
    return np.array(points[:n])

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # 10s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            # Use the provided relax function
            e, pos = relax(init_positions, potential, None)
            if e < best_energy and np.all(np.isfinite(pos)):
                best_energy = e
                best_positions = pos
        except Exception:
            pass

    # Generate diverse seeds
    seeds = []
    
    # Seed 1: Mackay Icosahedron (defective for n != 55)
    try:
        seeds.append(_generate_mackay_icosahedron(size))
    except Exception:
        pass
        
    # Seed 2: Growing Icosahedron
    try:
        seeds.append(_generate_growing_icosahedron(size, seed=1))
    except Exception:
        pass
        
    # Seed 3: Distorted Mackay
    try:
        base = _generate_mackay_icosahedron(size)
        rng = np.random.default_rng(2)
        distorted = base + rng.normal(0, 0.1, base.shape)
        seeds.append(distorted)
    except Exception:
        pass
        
    # Seed 4: Another Growing Icosahedron with different seed
    try:
        seeds.append(_generate_growing_icosahedron(size, seed=100))
    except Exception:
        pass
        
    # Seed 5: Random compact (fallback)
    try:
        rng = np.random.default_rng(3)
        radius = 1.5 * size ** (1.0/3.0)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 200:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 1.0:
                    continue
            points.append(p)
            attempts += 1
        if len(points) < size:
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
        seeds.append(np.array(points[:size]))
    except Exception:
        pass

    # Relax all seeds
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        if seed is not None and len(seed) == size:
            try_relax(seed, label=f"seed_{i}")

    # Basin Hopping / Perturbation on the best found
    if best_positions is not None and time.time() < deadline - 20:
        for attempt in range(5):
            if time.time() > deadline:
                break
            try:
                rng = np.random.default_rng(1000 + attempt)
                amp = 0.1 + 0.05 * attempt
                perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                try_relax(perturbed, label=f"perturb_{attempt}")
            except Exception:
                pass

    # Final fallback
    if best_positions is None:
        best_positions = _generate_mackay_icosahedron(size)

    return best_positions
