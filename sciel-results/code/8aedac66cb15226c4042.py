import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize
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

def _get_icosahedron_edges_and_faces():
    """
    Get edges and faces of the icosahedron defined by _get_icosahedron_vertices.
    """
    verts = _get_icosahedron_vertices(1.0)
    
    # Find edges (shortest distances)
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist, atol=1e-10):
                edges.append((i, j))
    
    # Find faces (triangles of edges)
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if ((min(i,j), max(i,j)) in edge_set and
                    (min(j,k), max(j,k)) in edge_set and
                    (min(i,k), max(i,k)) in edge_set):
                    faces.append((i, j, k))
    
    return edges, faces

def _build_mackay_icosahedron(n, r1=1.0, r2=1.75):
    """
    Build a Mackay icosahedral cluster of size n.
    For n=55: 1 center + 12 shell1 + 42 shell2
    For other sizes: remove atoms from the outer shell systematically.
    """
    center = np.array([0.0, 0.0, 0.0])
    verts_unit = _get_icosahedron_vertices(1.0)
    edges, faces = _get_icosahedron_edges_and_faces()
    
    # Shell 1: 12 vertices at radius r1
    shell1 = verts_unit * r1
    
    # Build shell 2 directions: 12 vertices + 20 face centers + 10 edge midpoints = 42
    shell2_dirs = []
    
    # 12 vertex directions
    for v in verts_unit:
        shell2_dirs.append(v)
    
    # 20 face centers
    for face in faces:
        fc = (verts_unit[face[0]] + verts_unit[face[1]] + verts_unit[face[2]]) / 3.0
        norm = np.linalg.norm(fc)
        if norm > 1e-10:
            shell2_dirs.append(fc / norm)
    
    # 10 edge midpoints (pick 10 out of 30)
    edge_dirs = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            edge_dirs.append(mid / norm)
    
    # Take first 10 edge midpoints
    for d in edge_dirs[:10]:
        shell2_dirs.append(d)
    
    shell2 = np.array(shell2_dirs) * r2
    
    # Full 55-atom structure
    full_55 = np.vstack([center, shell1, shell2])
    
    if n == 55:
        return full_55
    elif n == 13:
        return np.vstack([center, shell1])
    elif n == 1:
        return center.reshape(1, 3)
    else:
        # Remove atoms from the outer shell (indices 13-54)
        # Prefer removing edge midpoints first (indices 43-52), then face centers (25-42), then vertices (13-24)
        num_remove = 55 - n
        
        # Build removal order: edge midpoints first, then face centers, then vertices
        removal_order = []
        # Edge midpoints: indices 43 to 52
        for idx in range(52, 42, -1):
            removal_order.append(idx)
        # Face centers: indices 25 to 42
        for idx in range(42, 24, -1):
            removal_order.append(idx)
        # Vertices: indices 13 to 24
        for idx in range(24, 12, -1):
            removal_order.append(idx)
        
        to_remove = set(removal_order[:num_remove])
        keep_mask = np.ones(55, dtype=bool)
        for idx in to_remove:
            keep_mask[idx] = False
        
        return full_55[keep_mask]

def _check_min_distance(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True

def _optimize_with_lbfgs(positions, time_limit=15.0, maxiter=500):
    """
    Optimize using L-BFGS-B with Dzugutov gradient.
    """
    start_time = time.time()
    x0 = positions.flatten()
    
    def fun(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        pos = x.reshape(-1, 3)
        try:
            E, G = dzugutov_energy_gradient(pos)
            return E, G.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)
    
    res = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
    
    x_opt = res.x.reshape(-1, 3)
    return x_opt

def _relax_and_check(positions, potential, deadline):
    """
    Try to relax the structure and check if it's valid.
    Returns (energy, positions) if successful, (inf, None) otherwise.
    """
    try:
        energy, relaxed_pos = relax(positions, potential, None)
        if energy < 1e9 and _check_min_distance(relaxed_pos):
            return energy, relaxed_pos
    except Exception:
        pass
    return float('inf'), None

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
    
    def update_best(pos):
        nonlocal best_energy, best_positions
        try:
            E, _ = dzugutov_energy_gradient(pos)
            if E < best_energy and _check_min_distance(pos):
                best_energy = E
                best_positions = pos.copy()
        except Exception:
            pass
    
    # Generate initial structures
    init_structures = []
    
    # 1. Mackay icosahedron template
    mackay = _build_mackay_icosahedron(size, r1=1.0, r2=1.75)
    init_structures.append(mackay)
    
    # 2. Scaled versions
    for scale in [0.9, 1.05, 1.1]:
        init_structures.append(mackay * scale)
    
    # 3. Perturbed versions
    rng = np.random.default_rng(size + 42)
    for amp in [0.05, 0.1, 0.2]:
        perturbed = mackay + rng.normal(0, amp, mackay.shape)
        init_structures.append(perturbed)
    
    # 4. Random compact structure
    radius = 1.2 * size ** (1.0/3.0)
    points = []
    attempts = 0
    while len(points) < size and attempts < size * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.55:
                continue
        points.append(p)
        attempts += 1
    if len(points) < size:
        while len(points) < size:
            points.append(rng.uniform(-radius, radius, 3))
    init_structures.append(np.array(points))
    
    # 5. FCC-based structure
    fcc_points = []
    a_fcc = 1.5
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p = np.array([i, j, k]) * a_fcc
                if np.linalg.norm(p) < radius * 1.2:
                    fcc_points.append(p)
                if len(fcc_points) >= size:
                    break
            if len(fcc_points) >= size:
                break
        if len(fcc_points) >= size:
            break
    if len(fcc_points) >= size:
        init_structures.append(np.array(fcc_points[:size]))
    
    # Optimize each initial structure
    for i, init_pos in enumerate(init_structures):
        if time.time() > deadline - 10:
            break
        
        # Strategy 1: Use the suite's relax function
        energy, relaxed = _relax_and_check(init_pos, potential, deadline)
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
        
        # Strategy 2: L-BFGS-B from the relaxed structure
        if time.time() < deadline - 15:
            try:
                start_pos = best_positions if best_positions is not None and i > 0 else init_pos
                opt_pos = _optimize_with_lbfgs(start_pos, time_limit=12.0, maxiter=300)
                update_best(opt_pos)
            except Exception:
                pass
        
        # Strategy 3: L-BFGS-B from the original init
        if time.time() < deadline - 15 and i > 0:
            try:
                opt_pos = _optimize_with_lbfgs(init_pos, time_limit=10.0, maxiter=200)
                update_best(opt_pos)
            except Exception:
                pass
    
    # If we still have time, do additional perturbation-based search
    if best_positions is not None and time.time() < deadline - 20:
        rng2 = np.random.default_rng(size + 999)
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            perturbed = best_positions + rng2.normal(0, 0.08, best_positions.shape)
            try:
                opt_pos = _optimize_with_lbfgs(perturbed, time_limit=10.0, maxiter=150)
                update_best(opt_pos)
            except Exception:
                pass
    
    if best_positions is None:
        # Fallback: return the first initial structure
        return init_structures[0]
    
    return best_positions
