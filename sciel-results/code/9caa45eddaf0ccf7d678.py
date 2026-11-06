"""
Strategy: Lamarckian Basin Hopping with Global Rigid-Body Perturbations and Multi-Start Parallelism.

The Dzugutov potential favors polytetrahedral order. The current approach uses local SA which is slow.
We switch to:
1. Generating diverse polytetrahedral initializations (Mackay Icosahedra, deformed).
2. Using a pool of workers to run independent Basin Hopping chains in parallel.
3. Each BH chain uses rigid body transformations (rotation/translation) as the perturbation step, 
   which is physically more appropriate for clusters than moving single atoms.
4. After perturbation, we relax the structure.
5. We keep the global best structure found across all workers.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 10s safety margin for 300s limit (evaluator kills at 330s, but task says 300s)
# The prompt says "all calls together within 300 s". Let's be safe with 280s internal limit.
TIME_LIMIT = 280.0

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_mackay_icosahedron(n, scale=1.0):
    """
    Generates a Mackay icosahedral cluster.
    Magic numbers: 1, 13, 55, 147, 309...
    For n in [38, 45, 50, 55, 57, 60], we construct the 55-atom cluster and trim or add.
    The 55-atom cluster consists of:
    - 1 central atom
    - 12 first shell atoms (vertices of icosahedron)
    - 42 second shell atoms (12 face centers, 30 edge midpoints? No, Mackay 55 is 1+12+42)
      Actually, the second shell of a Mackay icosahedron has 42 atoms:
      - 12 atoms at the centers of the 20 faces? No.
      - The 55-atom Mackay icosahedron is formed by 13 interpenetrating 13-atom icosahedra? 
      - Simpler construction: 
        Shell 1: 12 vertices of regular icosahedron.
        Shell 2: 30 edge midpoints of the icosahedron + 12 face centers? 
        Let's check distances.
        In a regular icosahedron with edge length a, the distance from center to vertex is R = a/2 * sqrt(10+2*sqrt(5)).
        The 55-atom cluster usually has atoms at:
        1. Center
        2. 12 vertices of icosahedron (radius R1)
        3. 30 midpoints of edges (radius R2)
        4. 12 centers of faces (radius R3)
        Wait, 1+12+30+12 = 55. Yes.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of regular icosahedron
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Scale such that nearest neighbor distance is approx 1.8 (equilibrium for Dzugutov)
    # Min distance in unit icosahedron is 2.
    # So scale = 1.8 / 2 = 0.9
    R1 = 0.9 * scale
    
    points = [np.zeros(3)]
    if n > 1:
        points.extend([v * R1 for v in verts])
        
    # If n > 13, add second shell
    if n > 13:
        # Edge midpoints
        # Icosahedron has 30 edges.
        # We need to identify edges. Two vertices are connected if their distance is 2 (in unit sphere).
        # In our scaled verts, distance is 2 * R1.
        # Let's find all pairs with distance approx 2*R1.
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if np.isclose(d, 2.0, atol=1e-6):
                    edges.append((i, j))
        
        # 30 edge midpoints
        edge_mids = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            # Normalize? No, the midpoint of a chord in a unit sphere is at distance sqrt(1 - (d/2)^2) from center?
            # Distance from center to midpoint of edge in unit sphere:
            # The triangle formed by center and two vertices is isosceles.
            # Base = 2. Height = sqrt(R^2 - (Base/2)^2) = sqrt(1 - 1) = 0? 
            # Wait, for unit sphere, distance between adjacent vertices is 2.
            # So the midpoint is at distance 0 from center? No.
            # Let's re-calculate.
            # Verts are on unit sphere.
            # Dot product of adjacent verts:
            # v1 = [0, 1, phi], v2 = [1, phi, 0]
            # dot = phi.
            # cos(theta) = phi. 
            # Wait, phi > 1. This is impossible for unit vectors.
            # Ah, the vertices [0, 1, phi] are NOT on a unit sphere.
            # Norm is sqrt(1 + phi^2) = sqrt(1 + phi + 1) = sqrt(phi + 2).
            # So we must normalize first.
            # My code above does: verts = verts / norms. So they are on unit sphere.
            # Distance between adjacent vertices on unit sphere:
            # d^2 = |v1 - v2|^2 = 2 - 2 dot(v1, v2).
            # dot(v1, v2) for [0,1,phi]/norm and [1,phi,0]/norm.
            # Unnormalized dot: phi.
            # Norm^2 = 1 + phi^2 = phi + 2.
            # Normalized dot = phi / (phi + 2).
            # phi + 2 = 1.618 + 2 = 3.618.
            # phi / 3.618 = 0.4472.
            # d^2 = 2 - 2(0.4472) = 2 - 0.8944 = 1.1056.
            # d = 1.0514.
            # So min distance in unit icosahedron is ~1.05.
            # Previously I thought it was 2. That was wrong because I didn't normalize properly in the thought process.
            # Let's re-scale.
            # We want min dist ~ 1.8.
            # Scale factor S = 1.8 / 1.0514 = 1.712.
            
            # Let's just compute the min distance of the normalized verts and scale accordingly.
            pass
            
        # Recompute scale properly
        min_d = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < min_d:
                    min_d = d
        R1 = 1.8 / min_d * scale
        
        # Regenerate first shell with correct R1
        points = [np.zeros(3)]
        points.extend([v * R1 for v in verts])
        
        # Now add edge midpoints
        # The midpoints of the chords connecting adjacent vertices.
        # In the scaled system, the vertices are at distance R1 from center.
        # The angle between adjacent vertices is theta where cos(theta) = min_d/2? No.
        # In unit sphere, chord length L = 2 sin(theta/2).
        # min_d (unit) = L.
        # The distance from center to midpoint of chord is R * cos(theta/2) = sqrt(1 - (L/2)^2).
        # So R2_unit = sqrt(1 - (min_d/2)^2).
        R2_unit = np.sqrt(1 - (min_d/2)**2)
        R2 = R2_unit * R1
        
        for i, j in edges:
            mid_unit = (verts[i] + verts[j]) / 2.0
            # Normalize mid_unit? It's already at distance R2_unit from origin.
            mid_scaled = mid_unit * (R1 / 1.0) # Since verts are unit, mid_unit is just the geometric midpoint in unit space.
            # Wait, verts are unit vectors. The midpoint of the chord is (v1+v2)/2.
            # Its length is R2_unit.
            # We want it at distance R2.
            # So we scale by R2 / R2_unit.
            mid_scaled = mid_unit * (R2 / R2_unit)
            points.append(mid_scaled)
            
        # If n > 43, add face centers
        if n > 43:
            # Find faces. 20 faces.
            # A face is defined by 3 mutually connected vertices.
            # We can find them by checking triangles of vertices.
            faces = []
            for i in range(12):
                for j in range(i+1, 12):
                    for k in range(j+1, 12):
                        # Check if i-j, j-k, i-k are edges
                        if (i,j) in edges or (j,i) in edges:
                            if (j,k) in edges or (k,j) in edges:
                                if (i,k) in edges or (k,i) in edges:
                                    faces.append((i,j,k))
                                    
            # 20 faces
            # Distance from center to face center in unit sphere:
            # For regular icosahedron, inradius r = a/4 * sqrt(250+110*sqrt(5))?
            # Simpler: The face center is the average of the 3 vertices.
            # Let's compute the norm of the average of 3 adjacent unit vectors.
            # It should be constant.
            R3_unit = np.linalg.norm((verts[faces[0][0]] + verts[faces[0][1]] + verts[faces[0][2]]) / 3.0)
            R3 = (R3_unit / R2_unit) * R2 # Scale relative to R2? No, scale relative to R1.
            # R3 is a fixed fraction of the circumradius R1.
            # R3 = R1 * (R3_unit / 1.0) ? No, verts are unit, so R1 is the scale factor applied to unit vectors.
            # So R3 = R1 * R3_unit.
            R3 = R1 * R3_unit
            
            for i, j, k in faces:
                center_unit = (verts[i] + verts[j] + verts[k]) / 3.0
                center_scaled = center_unit * (R3 / R3_unit) # center_unit has norm R3_unit. We want norm R3.
                points.append(center_scaled)
                
    # If we have more points than n, we need to remove some.
    # If we have fewer, we need to add some.
    # For n in [38, 60], the 55-atom cluster is the target.
    # If n < 55, we can remove atoms from the outermost shell (face centers first, then edge midpoints).
    # If n > 55, we add atoms randomly on a larger shell.
    
    # Sort points by distance from center to identify shells
    dists = np.linalg.norm(np.array(points), axis=1)
    order = np.argsort(dists)
    points = [points[i] for i in order]
    
    if len(points) > n:
        # Remove from the end (outermost)
        points = points[:n]
    elif len(points) < n:
        # Add random points on a sphere of radius R1 * 1.5
        R_add = R1 * 1.5
        while len(points) < n:
            # Random point on sphere
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.uniform(-1, 1)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * R_add
            # Check min distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
                
    return np.array(points)

def _generate_deformed_icosahedron(n, scale=1.0, deformation=0.1):
    """Generate an icosahedron with random deformations."""
    pos = _generate_mackay_icosahedron(n, scale)
    # Apply small random displacements
    noise = np.random.normal(0, deformation, pos.shape)
    pos += noise
    return pos

def _generate_bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster."""
    points = []
    # BCC lattice points
    # Lattice constant a such that nearest neighbor is ~1.8.
    # NN in BCC is sqrt(3)/2 * a.
    # 1.8 = 0.866 * a => a = 2.078.
    a = 2.078 * scale
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p1 = np.array([i, j, k]) * a
                if len(points) < n and all(np.linalg.norm(p1 - q) >= 0.5 for q in points):
                    points.append(p1)
                p2 = np.array([i+0.5, j+0.5, k+0.5]) * a
                if len(points) < n and all(np.linalg.norm(p2 - q) >= 0.5 for q in points):
                    points.append(p2)
    if len(points) < n:
        while len(points) < n:
            p = np.random.uniform(-3, 3, 3) * scale
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points[:n])

def _relax_and_check(pos, potential, parameter):
    try:
        e, pos_relaxed = relax(pos, potential, parameter)
        # Check min distance
        min_d = np.inf
        for i in range(len(pos_relaxed)):
            for j in range(i+1, len(pos_relaxed)):
                d = np.linalg.norm(pos_relaxed[i] - pos_relaxed[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.5:
            return np.inf, pos_relaxed
        return e, pos_relaxed
    except Exception:
        return np.inf, pos

def _basin_hopping_worker(start_pos, potential, parameter, time_limit, seed):
    """
    Runs a single Basin Hopping chain.
    Returns the best energy and positions found.
    """
    np.random.seed(seed)
    pos = start_pos.copy()
    e, pos = _relax_and_check(pos, potential, parameter)
    best_e = e
    best_pos = pos.copy()
    
    T = 1.0
    cooling_rate = 0.95
    step_size = 0.5
    
    while time.time() - _START_TIME < time_limit:
        # Perturb: rotate and translate
        # Rotation
        v = np.random.normal(size=3)
        v /= np.linalg.norm(v)
        angle = np.random.uniform(0, 2*np.pi) * step_size
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + s*K + (1-c)*(K@K)
        
        # Translation
        t = np.random.normal(0, step_size, 3)
        
        new_pos = (pos @ R.T) + t
        
        # Relax
        new_e, new_pos_relaxed = _relax_and_check(new_pos, potential, parameter)
        
        delta_e = new_e - e
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_relaxed
            e = new_e
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
                
        T *= cooling_rate
        if T < 0.01:
            break
            
    return best_e, best_pos

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse initializations
    initializers = [
        lambda: _generate_mackay_icosahedron(size, 1.0),
        lambda: _generate_mackay_icosahedron(size, 0.95),
        lambda: _generate_mackay_icosahedron(size, 1.05),
        lambda: _generate_deformed_icosahedron(size, 1.0, 0.2),
        lambda: _generate_deformed_icosahedron(size, 1.0, 0.5),
        lambda: _generate_bcc_cluster(size, 1.0),
        lambda: _generate_bcc_cluster(size, 0.9),
    ]
    
    # Pre-relax initializations to get valid starting points
    candidates = []
    for i, init_func in enumerate(initializers):
        if not _check_time():
            break
        try:
            pos = init_func()
            e, pos_relaxed = _relax_and_check(pos, potential, parameter)
            if e < np.inf:
                candidates.append((e, pos_relaxed, i))
        except Exception:
            continue
            
    if not candidates:
        # Fallback
        pos = np.random.uniform(-2, 2, (size, 3))
        e, pos_relaxed = _relax_and_check(pos, potential, parameter)
        return pos_relaxed

    # Sort by energy and pick top few for parallel BH
    candidates.sort(key=lambda x: x[0])
    top_candidates = [c[1] for c in candidates[:4]] # Use 4 threads
    
    best_energy = min([c[0] for c in candidates])
    best_positions = candidates[0][1]
    
    # Run parallel Basin Hopping
    # We have 4 threads available.
    # We will run one BH chain per candidate in parallel.
    # Time limit for workers: leave 10s for collection.
    worker_time_limit = time.time() + (TIME_LIMIT - (time.time() - _START_TIME) - 10)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, start_pos in enumerate(top_candidates):
            # Pass a unique seed
            seed = 42 + i
            futures.append(executor.submit(_basin_hopping_worker, start_pos, potential, parameter, worker_time_limit, seed))
            
        for future in concurrent.futures.as_completed(futures):
            if not _check_time():
                break
            try:
                e, pos = future.result(timeout=1)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
                
    return best_positions
# EVOLVE-BLOCK-END
