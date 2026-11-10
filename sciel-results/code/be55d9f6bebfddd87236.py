"""
Dzugutov cluster global minima search.
Combines structural heuristics (Icosahedral, FCC, Random) with parallel relaxation.
Uses a global deadline to ensure completion within 300s for 6 instances.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global time tracking
_START_TIME = None
_DEADLINE_OFFSET = 285.0  # Leave 15s margin for 300s limit

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE_OFFSET

def _generate_icosahedral(n, rng):
    """
    Generate an icosahedral-like cluster.
    13 atoms (center + 12 vertices) is a strong candidate for Dzugutov.
    For larger n, add atoms on the surface or in second shells.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices
    verts = [
        (-1,  phi,  0), ( 1,  phi,  0), (-1, -phi,  0), ( 1, -phi,  0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi,  0, -1), ( phi,  0,  1), (-phi,  0, -1), (-phi,  0,  1)
    ]
    verts = np.array(verts, dtype=float)
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    # Equilibrium distance for Dzugutov is around 2.0-2.2.
    # Let's use 2.1 as a starting guess for the shell radius.
    r_shell = 2.1
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Add 12 vertices
    for v in verts:
        if len(points) >= n:
            break
        points.append(v * r_shell)
        
    if n <= 13:
        return np.array(points[:n])
    
    # For n > 13, add atoms to fill the cluster.
    # Strategy: Add atoms at edge midpoints (13 + 30 = 43) and then random surface points.
    # Or simply random points in a sphere with rejection sampling to maintain density.
    
    # Let's try adding edge midpoints first for better local order.
    # Identify edges: pairs of vertices with distance close to edge length.
    # Edge length of icosahedron with circumradius R is 2*R*sin(pi/5) approx.
    # With R=1, edge is ~1.05.
    verts_scaled = verts * r_shell
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_scaled[i] - verts_scaled[j])
            # Edge length is the minimum distance between vertices
            if d < 1.2 * r_shell: # Heuristic
                edges.append((i, j))
                
    # Sort edges by length to get the true 30 edges
    edges_with_len = [(np.linalg.norm(verts_scaled[i] - verts_scaled[j]), i, j) for i, j in edges]
    edges_with_len.sort()
    # Take the shortest 30 (or as many as fit)
    # Actually, there are exactly 30 edges.
    # Let's just take all that are "close" to the min distance.
    min_dist = edges_with_len[0][0] if edges_with_len else 1.0
    real_edges = [(i, j) for d, i, j in edges_with_len if d < min_dist * 1.1]
    
    for i, j in real_edges:
        if len(points) >= n:
            break
        mid = (verts_scaled[i] + verts_scaled[j]) / 2.0
        # Check if it's too close to existing points
        if all(np.linalg.norm(mid - p) > 0.5 for p in points):
            points.append(mid)
            
    # If still not enough, add random points in the volume or on surface
    # For Dzugutov, surface tension is low, so bulk-like filling is good.
    # Let's add random points in a sphere of radius r_shell * 1.5
    R_max = r_shell * 1.6
    while len(points) < n:
        # Random in sphere
        u = rng.random(3)
        r = R_max * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        # Minimum distance constraint
        if all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
        else:
            # Try jittering
            p_jittered = p + rng.normal(scale=0.1, size=3)
            if all(np.linalg.norm(p_jittered - q) >= 0.6 for q in points):
                points.append(p_jittered)
            # If still fails, skip this iteration (rejection sampling)
            # But to avoid infinite loops, we might force add if we've tried too many times
            # For now, just loop. The loop condition is len(points) < n.
            # To prevent infinite loop, we can break after many attempts and force add.
            # But simple rejection should work for reasonable n.
            
    # Jitter all points slightly to break symmetry and help relaxation
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.02, size=points.shape)
    
    return points

def _generate_fcc(n, rng):
    """
    Generate an FCC-like cluster.
    """
    a = 2.0 # Lattice constant
    # Generate FCC points in a box
    points = []
    R = (3.0 * n / (4.0 * np.pi) * (a**3 / 4)) ** (1.0 / 3.0) * 1.2
    box_size = 2 * R + a
    nx = int(box_size / a) + 2
    ny = int(box_size / a) + 2
    nz = int(box_size / a) + 2
    
    center = np.array([box_size/2, box_size/2, box_size/2])
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i, j, k]) * a
                # FCC basis
                p1 = base + np.array([0, 0, 0])
                p2 = base + np.array([0.5, 0.5, 0]) * a
                p3 = base + np.array([0.5, 0, 0.5]) * a
                p4 = base + np.array([0, 0.5, 0.5]) * a
                
                for p in [p1, p2, p3, p4]:
                    # Center the cluster
                    p_c = p - center
                    if np.linalg.norm(p_c) <= R:
                        points.append(p_c)
                        
    if len(points) < n:
        # Fallback: add random points
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
            elif len(points) < n and rng.random() < 0.01: # Force add occasionally
                points.append(p)
    else:
        # Shuffle and take n
        idx = rng.choice(len(points), n, replace=False)
        points = [points[i] for i in idx]
        
    points = np.array(points)
    points = points + rng.normal(scale=0.02, size=points.shape)
    return points

def _generate_random(n, rng):
    """
    Generate a random compact cluster.
    """
    # Estimate radius
    # Volume per atom ~ 10-15 in reduced units?
    # Let's assume density similar to icosahedral.
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.1
    
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        # Uniform in sphere
        u = rng.random(3)
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
        attempts += 1
        
    if len(points) < n:
        # If we couldn't fit enough, expand R or force add
        # For now, just add random points anywhere to ensure we have n
        while len(points) < n:
            p = rng.uniform(-R*1.5, R*1.5, 3)
            points.append(p)
            
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.02, size=points.shape)
    return points

def _worker(args):
    """
    Worker function to relax a single starting structure.
    """
    positions, potential, parameter, deadline = args
    try:
        # Check time
        if time.time() > deadline - 5:
            return (np.inf, positions)
            
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception as e:
        return (np.inf, positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    deadline = _get_deadline()
    
    rng_seed = size
    rng = np.random.default_rng(rng_seed)
    
    # Generate diverse starting structures
    # 1. Icosahedral
    ico1 = _generate_icosahedral(size, rng)
    
    # 2. FCC
    fcc1 = _generate_fcc(size, rng)
    
    # 3. Random
    rand1 = _generate_random(size, rng)
    
    # 4. Second Icosahedral with different seed/jitter
    rng2 = np.random.default_rng(rng_seed + 123)
    ico2 = _generate_icosahedral(size, rng2)
    
    # 5. Second Random
    rng3 = np.random.default_rng(rng_seed + 456)
    rand2 = _generate_random(size, rng3)
    
    # Add more diversity
    rng4 = np.random.default_rng(rng_seed + 789)
    rand3 = _generate_random(size, rng4)
    
    rng5 = np.random.default_rng(rng_seed + 1011)
    ico3 = _generate_icosahedral(size, rng5)
    
    candidates = [ico1, fcc1, rand1, ico2, rand2, rand3, ico3]
    
    # Parallel relaxation
    # Use 4 threads as per contract
    tasks = []
    for pos in candidates:
        tasks.append((pos, potential, parameter, deadline))
        
    best_energy = np.inf
    best_positions = candidates[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 2:
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
    if remaining > 30 and best_energy < np.inf:
        try:
            energy2, relaxed2 = relax(best_positions, potential, parameter)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = relaxed2
        except Exception:
            pass
            
    return best_positions
