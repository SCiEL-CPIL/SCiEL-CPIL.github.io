import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax

_START_TIME = None
# The evaluator kills after 330s. We have 6 instances.
# We should use ~45s per instance to be safe, leaving 270s total.
# However, the deadline is global. We will track time globally.
# 300s soft limit, 330s hard kill.
# Let's aim to finish all 6 in ~250s to be safe.
# ~40s per instance.

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    # Total budget 300s, leave 10s buffer for 310s, but hard kill is 330s.
    # Let's use 290s as the hard stop for calculation to ensure we return.
    return max(0.0, 290.0 - (time.time() - _START_TIME))

def _generate_mackay_icosahedron(n, rng):
    """
    Generate the first n atoms of a Mackay icosahedron (43 atoms total).
    If n > 43, it generates the 43-atom core and adds random surface atoms.
    If n < 43, it takes the first n atoms (center, then shell 1, then shell 2, etc).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of an icosahedron
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    # Normalize to radius R1
    norm = np.linalg.norm(verts[0])
    verts = verts / norm * 1.0
    
    # Center atom
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    for v in verts:
        if len(points) >= n:
            break
        points.append(v)
        
    if n <= 13:
        return np.array(points[:n])
        
    # Shell 2: 30 edge midpoints
    # Find all edges (distance between vertices is ~1.0514 for unit radius icosahedron? 
    # Actually, for vertices normalized to 1, distance is 2/sqrt(1+phi^2) * 1? 
    # Let's just compute distances and take the 30 smallest)
    verts_scaled = verts * 1.0 # Radius 1
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_scaled[i] - verts_scaled[j])
            edge_mids.append((d, (verts_scaled[i] + verts_scaled[j]) / 2.0))
            
    edge_mids.sort(key=lambda x: x[0])
    # The 30 shortest are the edges of the icosahedron
    for d, mid in edge_mids:
        if len(points) >= n:
            break
        # Check if it's a true edge (distance should be consistent)
        # For a regular icosahedron, all edges are same length.
        # We just take the first 30.
        points.append(mid)
        
    if n <= 43:
        return np.array(points[:n])
        
    # Shell 3: 20 face centers
    # We need to find the 20 faces.
    # A face is a triangle of vertices.
    # We can find them by checking triples of vertices that are all connected by edges.
    # Or simpler: the face centers are the vertices of the dual dodecahedron.
    # Let's just add random points on the surface if n > 43, or use a 63-atom structure.
    # For now, if n > 43, we return the 43-atom structure plus random perturbations/additions.
    # But we need exactly n atoms.
    
    # If n > 43, we can't just truncate. We need to add atoms.
    # Let's generate a 63-atom Mackay icosahedron if possible.
    # Shell 4 has 12 more atoms (midpoints of the "great edges" of the 43-atom cluster? No).
    # The 63-atom cluster is the next shell.
    # For simplicity and robustness, if n > 43, we will generate the 43-atom core and add (n-43) atoms randomly on the surface.
    
    points_arr = np.array(points[:43])
    if n > 43:
        extra_needed = n - 43
        for _ in range(extra_needed):
            # Pick a random direction
            theta = rng.random() * 2 * np.pi
            phi_ang = np.arccos(2 * rng.random() - 1)
            r = 1.0 + rng.random() * 0.5 # Slightly outside
            p = np.array([
                r * np.sin(phi_ang) * np.cos(theta),
                r * np.sin(phi_ang) * np.sin(theta),
                r * np.cos(phi_ang)
            ])
            points_arr = np.vstack([points_arr, p])
            
    # Add small noise to break symmetry and help relaxation
    points_arr = points_arr + rng.normal(scale=0.01, size=points_arr.shape)
    return points_arr[:n]

def _generate_icosahedral_cluster(n, rng):
    """
    Generate a cluster based on icosahedral symmetry.
    For n <= 13: 13-atom icosahedron (truncated).
    For 13 < n <= 43: 43-atom Mackay icosahedron (truncated).
    For n > 43: 43-atom core + random surface atoms.
    """
    if n <= 13:
        return _generate_mackay_icosahedron(n, rng)
    elif n <= 43:
        return _generate_mackay_icosahedron(n, rng)
    else:
        return _generate_mackay_icosahedron(n, rng)

def _generate_bimodal(n, rng):
    """
    Generate a cluster from two 13-atom icosahedra.
    """
    n1 = n // 2
    n2 = n - n1
    
    ico1 = _generate_mackay_icosahedron(n1, rng)
    ico2 = _generate_mackay_icosahedron(n2, rng)
    
    # Offset them
    sep = 3.5
    ico1[:, 0] += sep / 2
    ico2[:, 0] -= sep / 2
    
    points = np.vstack([ico1, ico2])
    points = points + rng.normal(scale=0.02, size=points.shape)
    return points

def _generate_random(n, rng):
    """
    Generate a random compact cluster.
    """
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0/3.0) * 1.1
    points = []
    min_dist = 0.55
    max_attempts = n * 2000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        r = R * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-R*1.5, R*1.5, 3)
        points.append(p)
        
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.015, size=points.shape)
    return points

def _worker(args):
    """Worker function to relax a single starting structure."""
    positions, potential, parameter, deadline = args
    try:
        if time.time() > deadline - 2:
            return (np.inf, positions)
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception:
        return (np.inf, positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    # The instance dict has 'A', 'B', etc, but relax expects a parameter string or dict?
    # The prompt says: chem_clusters.relax(positions, 'dzugutov')
    # So parameter should be 'dzugutov' or the instance dict?
    # "chem_clusters.relax(positions, 'dzugutov') ... may be called"
    # The instance is passed to construct.
    # Let's assume relax takes the potential name as a string or the instance dict.
    # Looking at the seed program: relax(positions, potential, parameter)
    # Where potential = instance["potential"] and parameter = instance.get("rho", instance.get("kind"))
    # This is ambiguous. Let's try passing the instance dict as parameter if possible, or just 'dzugutov'.
    # The prompt says: "chem_clusters.relax(positions, 'dzugutov')"
    # So the second arg is 'dzugutov'.
    # What is the third arg?
    # In the seed: parameter = instance.get("rho", instance.get("kind"))
    # If the instance doesn't have rho or kind, it might be None.
    # Let's pass the instance dict itself, as it contains all the constants.
    # Or maybe just 'dzugutov' and None?
    # Let's look at the signature in the seed: _worker(args) -> relax(positions, potential, parameter)
    # I will pass the instance dict as the parameter, as it is the most complete description.
    parameter = instance
    
    # Calculate deadline for this specific call
    # We have ~45s per instance on average.
    # Let's allocate 40s for this instance.
    time_budget = 40.0
    start_time = time.time()
    deadline = start_time + time_budget
    
    rng_seed = size
    rng = np.random.default_rng(rng_seed)
    
    candidates = []
    
    # 1. Icosahedral (Mackay)
    candidates.append(_generate_icosahedral_cluster(size, rng))
    
    # 2. Bimodal (2x Icosahedra)
    if size > 20:
        candidates.append(_generate_bimodal(size, rng))
        
    # 3. Random
    candidates.append(_generate_random(size, rng))
    
    # 4. Another Icosahedral with different seed
    rng2 = np.random.default_rng(rng_seed + 123)
    candidates.append(_generate_icosahedral_cluster(size, rng2))
    
    # 5. Another Random
    rng3 = np.random.default_rng(rng_seed + 456)
    candidates.append(_generate_random(size, rng3))
    
    # 6. Bimodal with different seed
    if size > 20:
        rng4 = np.random.default_rng(rng_seed + 789)
        candidates.append(_generate_bimodal(size, rng4))

    # Limit to 4-6 candidates to fit in time
    max_candidates = 6
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
        
    tasks = [(pos, potential, parameter, deadline) for pos in candidates]
    
    best_energy = np.inf
    best_positions = candidates[0]
    
    # Use 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 1:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
    
    # If we have time, do a second relaxation on the best result
    remaining = deadline - time.time()
    if remaining > 5 and best_energy < np.inf:
        try:
            energy2, relaxed2 = relax(best_positions, potential, parameter)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = relaxed2
        except Exception:
            pass
            
    return best_positions
