import numpy as np
import time
import concurrent.futures
from scipy.spatial.transform import Rotation
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 310.0  # 20s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return __DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron-like cluster.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices)
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    Total 55 atoms.
    For n < 55, we take the first n atoms in a specific order.
    For n > 55, we add random atoms on a larger shell.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Scale factor for Dzugutov potential
    # Equilibrium distance is roughly around 1.8-2.0
    # Let's assume the first shell radius is R1 = 2.0
    R1 = 2.0
    
    points = []
    # Center
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: Vertices
    if n > 1:
        shell1 = verts * R1
        points.extend(shell1.tolist())
        
    # Shell 2: Edge midpoints and Face centers
    if n > 13:
        # Find edges (distance between verts is 2*R1/sqrt(1+phi^2)? No, just find closest)
        dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        edge_thresh = np.min(min_dists) * 1.1
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if dists[i, j] < edge_thresh:
                    edges.append((i, j))
                    
        # Edge midpoints (30)
        edge_mids = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            # Radius for shell 2 is R1 * sqrt(1 + (1/phi)^2)? 
            # Actually, distance from center to edge midpoint of icosahedron with radius R is R * cos(pi/5) * ... 
            # Let's just use the normalized vector and scale it.
            # The ratio of edge-mid radius to vertex radius is cos(pi/5) / cos(pi/10)? No.
            # For regular icosahedron, r_edge_mid / r_vertex = sqrt(1 - (1/4)*(1/phi^2))? 
            # Let's just calculate it: 
            # Vertex (0,1,phi). Norm = sqrt(1+phi^2). 
            # Edge mid (0,0,phi). Norm = phi.
            # Ratio = phi / sqrt(1+phi^2) = sqrt(phi) / sqrt(1+phi^2) * sqrt(phi)? 
            # phi^2 = phi+1. 1+phi^2 = phi+2.
            # Ratio = phi / sqrt(phi+2).
            ratio = phi / np.sqrt(phi + 2)
            R2 = R1 * ratio
            edge_mids.append(mid * R2)
            
        # Face centers (20)
        # Need to find faces. A face is a triangle of 3 vertices.
        # In icosahedron, each face is an equilateral triangle.
        faces = []
        # Iterate over all triplets? 12 choose 3 = 220. Check if all 3 are edges.
        for i in range(12):
            for j in range(i+1, 12):
                if dists[i, j] >= edge_thresh: continue
                for k in range(j+1, 12):
                    if dists[i, k] < edge_thresh and dists[j, k] < edge_thresh:
                        faces.append((i, j, k))
                        
        face_centers = []
        for i, j, k in faces:
            center = (verts[i] + verts[j] + verts[k]) / 3.0
            center /= np.linalg.norm(center)
            # Radius for face center
            # Vertex (0,1,phi). Face (0,1,phi), (1,phi,0), (phi,0,1) -> Center (1/3, (1+phi)/3, (phi+1)/3)
            # This is getting complicated. Let's estimate R3.
            # R3 is slightly larger than R2.
            # Ratio R3/R1 is approx 0.95? 
            # Actually, for Mackay icosahedron, the second shell is a truncated icosahedron? No, it's a specific set.
            # Let's just use R2 * 1.05 or similar.
            R3 = R1 * 0.95 # Approximation
            face_centers.append(center * R3)
            
        # Add shell 2 points
        # We need to add them in a way that fills up to n.
        # Order: maybe interleave? Or just add all and take first n?
        # Taking first n might bias towards edges.
        # Let's add all edge mids, then face centers.
        points.extend(edge_mids)
        points.extend(face_centers)
        
    # If n > 55, add random points on a larger shell
    if n > 55:
        needed = n - len(points)
        R4 = R1 * 1.5
        for _ in range(needed):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            p = [np.sin(phi_angle)*np.cos(theta), np.sin(phi_angle)*np.sin(theta), np.cos(phi_angle)]
            points.append(np.array(p) * R4)
            
    return np.array(points[:n])

def _generate_fcc(n, rng):
    """Generate FCC-like cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    a = 2.0
    max_coord = int(np.ceil(R / a)) + 2
    candidates = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    if x*x + y*y + z*z <= R*R:
                        candidates.append([x, y, z])
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return np.array([rng.uniform(-1, 1, 3) for _ in range(n)])
    if len(candidates) > n:
        dists = np.sum(candidates**2, axis=1)
        idx = np.argsort(dists)[:n]
        candidates = candidates[idx]
    elif len(candidates) < n:
        while len(candidates) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                candidates = np.vstack([candidates, p])
    return candidates[:n]

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    # Fill if failed
    while len(points) < n:
        p = rng.uniform(-1, 1, 3)
        points.append(p)
    return np.array(points[:n])

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def _assemble_subclusters(subclusters, potential, parameter, rng):
    """
    Take a list of subcluster positions (each is an array of shape (n_i, 3)),
    and optimize their relative translations and rotations to form a larger cluster.
    This is a simplified version: we place them at random positions and relax the whole thing.
    A more advanced version would optimize the inter-cluster degrees of freedom first.
    Given time constraints, we will just place them in a spherical arrangement and relax.
    """
    # Calculate total atoms
    total_n = sum(len(sc) for sc in subclusters)
    
    # Initial placement: place subclusters on a sphere
    R = 2.0 * (total_n / 4.0) ** (1.0/3.0)
    n_sub = len(subclusters)
    
    # Generate directions for subclusters
    if n_sub == 1:
        dirs = [np.array([0.0, 0.0, 0.0])]
    else:
        # Simple spherical distribution
        angles = np.linspace(0, 2*np.pi, n_sub, endpoint=False)
        dirs = []
        for i in range(n_sub):
            theta = angles[i]
            # Alternate z to spread them out
            z = 0.0 if i % 2 == 0 else 0.5
            dirs.append([np.cos(theta), np.sin(theta), z])
        dirs = np.array(dirs)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        
    # Scale directions
    # Distance between subcluster centers
    # Heuristic: distance ~ 2 * radius of subcluster + 2
    max_sub_size = max(len(sc) for sc in subclusters)
    sub_radius = 2.0 * (max_sub_size / 4.0) ** (1.0/3.0)
    dist_scale = 2 * sub_radius + 1.0
    
    combined = []
    for i, sc in enumerate(subclusters):
        # Rotate subcluster randomly
        rot = Rotation.random(random_state=rng)
        sc_rot = rot.apply(sc)
        # Translate
        offset = dirs[i] * dist_scale
        combined.append(sc_rot + offset)
        
    # Concatenate
    full_cluster = np.vstack(combined)
    
    # Relax the full cluster
    energy, relaxed = relax(full_cluster, potential, parameter)
    return energy, relaxed

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy 1: Monolithic Seeds
    seeds = []
    # 2 Icosahedral
    for _ in range(2):
        seeds.append(_generate_mackay_icosahedron(n, rng))
    # 1 FCC
    seeds.append(_generate_fcc(n, rng))
    # 2 Random
    for _ in range(2):
        seeds.append(_generate_random(n, rng))
        
    # Strategy 2: Divide and Conquer Assembly
    # We can try to assemble from smaller icosahedra if n allows.
    # e.g. for n=55, it's already one icosahedron.
    # for n=38, maybe 13 + 25? Or just relax.
    # Let's add a few assembly seeds.
    # Split n into two parts
    n1 = n // 2
    n2 = n - n1
    if n1 >= 1 and n2 >= 1:
        # Generate subclusters
        for _ in range(2):
            sc1 = _generate_mackay_icosahedron(n1, rng)
            sc2 = _generate_mackay_icosahedron(n2, rng)
            # Assemble
            try:
                energy, relaxed = _assemble_subclusters([sc1, sc2], potential, parameter, rng)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except:
                pass

    # Phase 1: Relax all monolithic seeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions

    if best_positions is None:
        # Fallback
        best_positions = seeds[0]
        best_energy = float('inf')
        try:
            best_energy, best_positions = relax(best_positions, potential, parameter)
        except:
            pass

    # Phase 2: Basin Hopping on the best structure
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    step_size = 0.3
    max_iters = 100
    
    for i in range(max_iters):
        if _get_time_remaining() < 10.0:
            break
            
        # Perturb
        noise = rng.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax
        try:
            energy, relaxed = relax(new_positions, potential, parameter)
        except:
            continue
            
        if energy < current_energy:
            current_energy = energy
            current_positions = relaxed
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
            # Decrease step size
            step_size *= 0.95
        else:
            # Metropolis accept?
            # For global min, we usually just reject or increase step size.
            # Let's increase step size to escape local minima
            step_size *= 1.05
            if step_size > 1.5:
                step_size = 1.5

    return best_positions
