# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.spatial import ConvexHull, Delaunay
import math

def _icosahedron_vertices():
    """Return vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
    return v

def _fcc_sphere(n):
    """
    Generate n atoms in an FCC-like spherical cluster.
    This is a better initialization than random for metallic clusters.
    """
    # Approximate radius for n atoms in FCC
    # Density ~ 4 atoms per unit cell volume (a^3). a ~ 1.
    # Volume ~ n/4. Radius ~ (3*n/4 / 4*pi)^(1/3)
    r_est = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.1
    
    points = []
    # Generate FCC lattice points within a sphere
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    max_idx = int(np.ceil(r_est)) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if np.linalg.norm(p) <= r_est:
                        points.append(p)
    
    if not points:
        return np.zeros((n, 3))
        
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    
    # If we have more than n, keep the ones closest to center
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Fill remaining with random points on the surface
        current_n = len(points)
        while current_n < n:
            # Random point on sphere of radius r_est
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            p = r_est * np.array([
                np.sin(phi)*np.cos(theta),
                np.sin(phi)*np.sin(theta),
                np.cos(phi)
            ])
            points = np.vstack([points, p])
            current_n += 1
            
    return points

def _hcp_sphere(n):
    """
    Generate n atoms in an HCP-like spherical cluster.
    """
    r_est = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.1 # Similar density
    points = []
    
    # HCP layers: ABAB...
    # Layer 0: Hexagonal grid
    # Layer 1: Offset
    # We just generate a hexagonal grid in 2D and stack them.
    
    max_idx = int(np.ceil(r_est)) + 2
    a = 1.0 # Lattice constant
    # Hexagonal spacing
    dx = a
    dy = a * np.sqrt(3) / 2
    
    layer_z = 0
    while len(points) < n * 1.2: # Generate a bit extra
        # Generate hexagonal grid for this layer
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                # Hexagonal offset
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                z = layer_z * a * np.sqrt(2/3) # Ideal HCP c/a ratio ~ 1.633 -> c = a*sqrt(2/3)*2? 
                # Actually c = 2 * sqrt(2/3) * a. So z step is c/2 = sqrt(2/3)*a.
                
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
        layer_z += 1
        
    if not points:
        return np.zeros((n, 3))
        
    points = np.array(points)
    points -= points.mean(axis=0)
    
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        current_n = len(points)
        while current_n < n:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            p = r_est * np.array([
                np.sin(phi)*np.cos(theta),
                np.sin(phi)*np.sin(theta),
                np.cos(phi)
            ])
            points = np.vstack([points, p])
            current_n += 1
            
    return points

def _icosahedral_shells(n):
    """
    Generate n atoms using icosahedral shells.
    Shell 1: 13 atoms (1 center + 12 vertices)
    Shell 2: 55 atoms (13 + 42)
    Shell 3: 147 atoms (55 + 92)
    """
    # We only go up to shell 2 or 3.
    # For n > 13, we start with 13 and add atoms.
    # For n > 55, we start with 55.
    
    v1 = _icosahedron_vertices()
    # Shell 1
    shell1 = np.vstack([np.zeros(3), v1])
    
    if n <= 13:
        # Just take first n of shell1 (center + some vertices)
        # This is a bad idea for n<13, but we assume n>=38
        return shell1[:n]
        
    # Shell 2: Add atoms at the midpoints of edges of the icosahedron? 
    # No, the 55-atom icosahedron has a specific structure.
    # It's easier to just use the FCC/HCP for larger n if we don't have exact templates.
    # But let's try to construct a rough 55-atom cluster.
    # The 55-atom cluster is a "fullerene-like" icosahedron.
    # Actually, let's just use the 13-atom core and grow it using a smart algorithm
    # if n is not exactly 13, 55, 147.
    
    # For now, if n is close to 55, use a modified 55-atom structure.
    # If n is close to 147, use 147.
    # Otherwise, fall back to FCC/HCP which are robust.
    
    # Since the task asks for n=38, 55, 59, 75, 79, 80.
    # 55 is a magic number.
    # Let's generate the 55-atom icosahedron explicitly.
    
    # 55 atom icosahedron:
    # 1 center
    # 12 vertices (radius 1)
    # 30 edge midpoints (radius ~ 0.724)
    # 12 face centers (radius ~ 0.525)
    # Total: 1 + 12 + 30 + 12 = 55.
    
    # We need the geometry of the icosahedron.
    # Vertices: v
    # Edges: connect vertices with distance d_edge.
    # Face centers: centroid of triangles.
    
    v = _icosahedron_vertices()
    
    # Find edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            # Icosahedron edge length is 2/sqrt(phi^2+1) * R? 
            # Min distance between vertices.
            edges.append((i, j, d))
    edges.sort(key=lambda x: x[2])
    # The 30 shortest distances are the edges.
    edge_len = edges[0][2]
    real_edges = [(i, j) for i, j, d in edges if np.isclose(d, edge_len)]
    
    edge_midpoints = []
    for i, j in real_edges:
        mid = 0.5 * (v[i] + v[j])
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
        
    # Find faces (triangles)
    # We can use Delaunay on the sphere or just check triples.
    # Simpler: The 20 faces of the icosahedron.
    # We can find them by looking for sets of 3 vertices that form a triangle with side length edge_len.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in real_edges and (j, i) not in real_edges:
                continue
            for k in range(j+1, 12):
                if (j, k) not in real_edges and (k, j) not in real_edges:
                    continue
                if (i, k) not in real_edges and (k, i) not in real_edges:
                    continue
                # Check if i, j, k form a face.
                # They must be mutually connected.
                faces.append((i, j, k))
                
    # There are 20 faces.
    face_centers = []
    for i, j, k in faces:
        c = (v[i] + v[j] + v[k]) / 3.0
        face_centers.append(c)
    face_centers = np.array(face_centers)
    
    # Combine
    # Scale radii to match a continuous structure?
    # The 55-atom cluster is not just these points scaled. It's a relaxed structure.
    # But as an initial guess, it's good.
    
    # Let's just return the 13-atom core if n < 55, or 55-atom if n >= 55?
    # No, we need exact n.
    
    # Strategy:
    # If n <= 13: return shell1[:n]
    # If 13 < n <= 55: Start with shell1, add atoms from edge_midpoints, then face_centers.
    # If n > 55: Start with 55-atom cluster, then add atoms on the surface of that.
    
    if n <= 13:
        return shell1[:n]
        
    candidates = [np.zeros(3)] + list(v) + list(edge_midpoints) + list(face_centers)
    candidates = np.array(candidates)
    
    # If n <= 55, just take the first n? 
    # The order in `candidates` is center, vertices, edges, faces.
    # This is a good hierarchical ordering.
    if n <= 55:
        return candidates[:n]
        
    # If n > 55, we have 55 atoms. We need to add more.
    # We can add atoms at the midpoints of the edges of the 55-atom cluster's convex hull?
    # Or just use FCC for n > 55 as it's more robust for larger sizes.
    # Given the time constraints and complexity, let's fall back to FCC for n > 55.
    # But wait, 59, 75, 79, 80 are the targets.
    # FCC is a very good guess for these sizes.
    
    return _fcc_sphere(n)

def _relax_worker(args):
    x, kind, n, seed_val = args
    try:
        e, p = relax(x, 'sutton_chen', kind)
        if not np.all(np.isfinite(p)) or not np.isfinite(e):
            return (np.inf, None)
        if n > 1:
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return (np.inf, None)
        return (e, p)
    except Exception:
        return (np.inf, None)

def _basin_hopping_worker(args):
    """
    Performs a basin hopping run starting from an initial structure.
    """
    x0, kind, n, seed_val, max_iters, time_limit = args
    
    rng = np.random.default_rng(seed_val)
    x = x0.copy()
    
    # Initial relaxation
    e, x = relax(x, 'sutton_chen', kind)
    if not np.all(np.isfinite(x)) or not np.isfinite(e):
        return (np.inf, None)
        
    best_e = e
    best_x = x.copy()
    
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() > start_time + time_limit:
            break
            
        # Kick: Random perturbation
        # Scale kick size with cluster size?
        # Typical kick: 0.1 to 0.5
        kick = 0.3
        x_kicked = x + rng.normal(0, kick, size=x.shape)
        
        # Relax kicked structure
        e_new, x_new = relax(x_kicked, 'sutton_chen', kind)
        
        if not np.all(np.isfinite(x_new)) or not np.isfinite(e_new):
            continue
            
        # Check validity
        if n > 1:
            d = np.sqrt(np.sum((x_new[:, None, :] - x_new[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                continue
        
        # Metropolis accept (T=1.0 usually, or just accept if better)
        # For global minima, we can just accept if better, or use Metropolis to escape.
        # Since we have multiple restarts, simple "accept if better" might get stuck.
        # Let's use Metropolis with T=0.5.
        delta_e = e_new - e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.5):
            x = x_new
            e = e_new
            
            if e < best_e:
                best_e = e
                best_x = x.copy()
                
    return (best_e, best_x)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    tasks = []
    
    # 1. Icosahedral Shells (Good for 55, maybe 38?)
    pos_ico = _icosahedral_shells(n)
    tasks.append((pos_ico, kind, n, 100))
    
    # 2. FCC Sphere
    pos_fcc = _fcc_sphere(n)
    tasks.append((pos_fcc, kind, n, 200))
    
    # 3. HCP Sphere
    pos_hcp = _hcp_sphere(n)
    tasks.append((pos_hcp, kind, n, 300))
    
    # 4. Random Compact (for diversity)
    pos_rand = np.random.default_rng(400).uniform(-1, 1, size=(n, 3))
    # Simple relaxation of random start? No, just pass to BH which relaxes first.
    tasks.append((pos_rand, kind, n, 400))
    
    best_e = np.inf
    best_p = None
    
    # Phase 1: Quick relaxation of initial guesses to find a good starting point for BH
    # We will run BH in parallel.
    # Each BH run has a time limit of ~60s.
    # We have 4 threads.
    
    max_iters_bh = 50 # Limit iterations to ensure time is managed
    time_limit_bh = 60.0
    
    bh_args = [(pos, kind, n, seed, max_iters_bh, time_limit_bh) for pos, kind, n, seed in tasks]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_basin_hopping_worker, arg) for arg in bh_args]
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 10.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
                
    # Phase 2: If we have time, do a second round of BH starting from the best found
    if best_p is not None and time.time() < deadline - 30.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 10.0:
            # Run 4 more BH runs starting from best_p with different seeds
            new_bh_args = []
            for i in range(4):
                seed = 1000 + i
                # Add a small perturbation to best_p to avoid identical starts
                rng = np.random.default_rng(seed)
                perturbed = best_p + rng.normal(0, 0.1, size=best_p.shape)
                new_bh_args.append((perturbed, kind, n, seed, 30, remaining_time/4.0))
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_basin_hopping_worker, arg) for arg in new_bh_args]
                for future in concurrent.futures.as_completed(futures):
                    if time.time() > deadline - 5.0:
                        break
                    try:
                        e, p = future.result()
                        if e < best_e:
                            best_e = e
                            best_p = p
                    except Exception:
                        pass
                        
    if best_p is None:
        # Fallback
        best_p = np.zeros((n, 3))
        for i in range(n):
            best_p[i] = [ (i % 10) * 0.5, ((i // 10) % 10) * 0.5, (i // 100) * 0.5 ]
            
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
