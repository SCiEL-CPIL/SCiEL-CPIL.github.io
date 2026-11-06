import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
MAX_TIME = 300.0

# Global time management
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None
_CALL_COUNT = 0

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_icosahedron_faces():
    """Return the 20 triangular faces of a regular icosahedron (indices into vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find all faces by checking which triplets of vertices form equilateral triangles
    # with the correct dihedral angle
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                v1 = verts[i]
                v2 = verts[j]
                v3 = verts[k]
                # Check if all three edges have the same length (icosahedron edge)
                e1 = np.linalg.norm(v1 - v2)
                e2 = np.linalg.norm(v2 - v3)
                e3 = np.linalg.norm(v3 - v1)
                # Icosahedron edge length is 2/sqrt(phi)
                expected_edge = 2.0 / np.sqrt(phi)
                if (abs(e1 - expected_edge) < 1e-10 and 
                    abs(e2 - expected_edge) < 1e-10 and 
                    abs(e3 - expected_edge) < 1e-10):
                    faces.append((i, j, k))
    return faces

def _get_icosahedron_edges():
    """Return the 30 edges of a regular icosahedron (pairs of vertex indices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edges = []
    expected_edge = 2.0 / np.sqrt(phi)
    for i in range(12):
        for j in range(i+1, 12):
            e = np.linalg.norm(verts[i] - verts[j])
            if abs(e - expected_edge) < 1e-10:
                edges.append((i, j))
    return edges

def _generate_proper_mackay_55():
    """
    Generate the proper 55-atom Mackay icosahedron with icosahedral symmetry.
    Structure: 1 center + 12 shell-1 (vertices) + 42 shell-2 atoms.
    Shell 2: 20 face centers + 12 vertices at larger radius + 10 edge midpoints? 
    Actually, the standard Mackay 55 has:
    - 1 center
    - 12 atoms at icosahedron vertices (shell 1)
    - 42 atoms in shell 2: these are the centers of the 20 faces and 
      additional atoms to maintain icosahedral symmetry.
    
    A cleaner construction: Use the fact that the 55-atom Mackay icosahedron
    can be built from nested icosahedra. Let's use a robust approach:
    1. Start with 13-atom icosahedron (center + 12 vertices)
    2. Add 42 atoms at positions that maintain icosahedral symmetry:
       - 20 atoms at face centers
       - 12 atoms at positions along the 12 vertex directions but at a different radius
       - 10 atoms at selected edge midpoints (to get exactly 42)
    
    Actually, let's use a simpler but effective method: generate atoms with 
    icosahedral symmetry by taking a base set and applying icosahedral rotations.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = _get_icosahedron_vertices()
    faces = _get_icosahedron_faces()
    edges = _get_icosahedron_edges()
    
    # Shell 1: 12 vertices at radius r1
    r1 = 1.0
    shell1 = verts * r1
    
    # Shell 2: We need 42 atoms with icosahedral symmetry
    # Option: Use face centers (20) + some edge midpoints + some additional
    # Face centers: centroid of each triangular face
    face_centers = []
    for face in faces:
        fc = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    # Normalize face centers to unit length, then scale to desired radius
    fc_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers_unit = face_centers / fc_norms
    r_face = 1.0 * np.sqrt(3) / 2.0  # Distance from center to face centroid for unit icosahedron
    # Actually, for unit icosahedron, face center distance is sqrt(3)/2 * (edge/sqrt(3))
    # Let's just normalize and use a radius
    r2 = 1.45
    shell2_faces = face_centers_unit * r2
    
    # Edge midpoints
    edge_mids = []
    for edge in edges:
        em = (verts[edge[0]] + verts[edge[1]]) / 2.0
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    em_norms = np.linalg.norm(edge_mids, axis=1, keepdims=True)
    edge_mids_unit = edge_mids / em_norms
    # We need 42 - 20 = 22 more atoms. Use 12 from vertex directions at different radius + 10 edge mids
    # Actually, let's use a different strategy: just use rejection sampling with icosahedral bias
    # to get exactly 42 atoms in shell 2
    
    # For now, use face centers (20) + rejection sample 22 more in shell 2
    rng = np.random.default_rng(42)
    shell2 = [shell2_faces.copy()]
    
    current_atoms = np.vstack([np.array([[0.0, 0.0, 0.0]]), shell1, shell2_faces])
    min_dist = 0.5
    added = 0
    attempts = 0
    max_attempts = 5000
    while added < 22 and attempts < max_attempts:
        # Sample on sphere of radius r2
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = np.array([
            r2 * sin_phi * np.cos(theta),
            r2 * sin_phi * np.sin(theta),
            r2 * cos_phi
        ])
        
        dists = np.linalg.norm(current_atoms - p, axis=1)
        if np.min(dists) >= min_dist:
            current_atoms = np.vstack([current_atoms, p])
            added += 1
        attempts += 1
    
    return current_atoms

def _generate_icosahedral_seed(n, rng):
    """
    Generate an icosahedral-symmetry-biased seed for n atoms.
    Uses nested icosahedra structure.
    """
    verts = _get_icosahedron_vertices()
    
    # Build up from 13-atom icosahedron
    atoms = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    r1 = 1.0
    atoms.extend(verts * r1)
    
    # If n > 13, add more shells
    if n > 13:
        # Shell 2: face centers and edge midpoints
        faces = _get_icosahedron_faces()
        edges = _get_icosahedron_edges()
        
        face_centers = []
        for face in faces:
            fc = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
            fc_norm = np.linalg.norm(fc)
            face_centers.append(fc / fc_norm)
        face_centers = np.array(face_centers)
        
        edge_mids = []
        for edge in edges:
            em = (verts[edge[0]] + verts[edge[1]]) / 2.0
            em_norm = np.linalg.norm(em)
            edge_mids.append(em / em_norm)
        edge_mids = np.array(edge_mids)
        
        r2 = 1.4
        # Add face centers first
        for fc in face_centers:
            if len(atoms) >= n:
                break
            atoms.append(fc * r2)
        
        # Add edge midpoints if needed
        for em in edge_mids:
            if len(atoms) >= n:
                break
            atoms.append(em * r2)
        
        # If still not enough, add more via rejection sampling
        if len(atoms) < n:
            atoms_arr = np.array(atoms)
            r3 = 1.8
            attempts = 0
            while len(atoms) < n and attempts < 10000:
                theta = rng.uniform(0, 2 * np.pi)
                cos_phi = rng.uniform(-1, 1)
                sin_phi = np.sqrt(1 - cos_phi**2)
                p = np.array([
                    r3 * sin_phi * np.cos(theta),
                    r3 * sin_phi * np.sin(theta),
                    r3 * cos_phi
                ])
                dists = np.linalg.norm(atoms_arr - p, axis=1)
                if np.min(dists) >= 0.5:
                    atoms.append(p)
                    atoms_arr = np.vstack([atoms_arr, p])
                attempts += 1
    
    return np.array(atoms[:n])

def _generate_random_compact_seed(n, rng):
    """Generate a random compact cluster seed."""
    radius = 0.8 * n ** (1/3) + 1.5
    points = []
    for _ in range(n):
        placed = False
        for attempt in range(500):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.5:
                    continue
            points.append(p)
            placed = True
            break
        if not placed:
            p = rng.uniform(-radius * 1.3, radius * 1.3, 3)
            points.append(p)
    return np.array(points)

def _generate_fcc_seed(n, rng):
    """Generate an FCC-based cluster seed."""
    # FCC lattice points
    points = []
    a = 1.0  # lattice parameter
    # Generate points in a sphere of appropriate radius
    radius = 0.8 * n ** (1/3) + 1.0
    max_coord = int(np.ceil(radius / a)) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: all corners + face centers
                # Corner
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= radius:
                    points.append(p)
                # Face centers
                for axis in range(3):
                    p_fc = np.array([i, j, k]) * a
                    p_fc[axis] += a / 2
                    if np.linalg.norm(p_fc) <= radius:
                        points.append(p_fc)
    
    points = np.array(points)
    if len(points) == 0:
        return _generate_random_compact_seed(n, rng)
    
    # Center the cluster
    points = points - np.mean(points, axis=0)
    
    # Select n closest to center
    dists = np.linalg.norm(points, axis=1)
    sort_idx = np.argsort(dists)
    if len(points) >= n:
        selected = points[sort_idx[:n]]
    else:
        # Need more points, add randomly
        selected = points.copy()
        while len(selected) < n:
            p = rng.normal(0, 0.5, 3) + selected[rng.integers(len(selected))]
            dists = np.linalg.norm(selected - p, axis=1)
            if np.min(dists) >= 0.5:
                selected = np.vstack([selected, p])
            else:
                selected = np.vstack([selected, p + rng.normal(0, 0.3, 3)])
    
    return selected

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE, _CALL_COUNT
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 295.0  # Safety margin
    
    _CALL_COUNT += 1
    size = instance["n"]
    start_time = time.time()
    
    # Time budget for this instance
    remaining = _GLOBAL_DEADLINE - start_time
    calls_left = 6 - _CALL_COUNT + 1
    instance_budget = min(remaining / calls_left, 50.0)
    instance_deadline = start_time + instance_budget
    
    rng = np.random.default_rng(size * 7919 + 12345)
    
    # Generate multiple diverse seeds
    seeds = []
    
    # Seed 1: Proper Mackay 55-based
    try:
        pos_55 = _generate_proper_mackay_55()
        e_55, pos_55_relaxed = relax(pos_55, POTENTIAL)
        
        if size <= 55:
            dists = np.linalg.norm(pos_55_relaxed, axis=1)
            sort_idx = np.argsort(dists)[::-1]
            num_to_remove = 55 - size
            remove_idx = sort_idx[:num_to_remove]
            keep_idx = np.setdiff1d(np.arange(55), remove_idx)
            pos_target = pos_55_relaxed[keep_idx]
        else:
            pos_target = pos_55_relaxed.copy()
            num_to_add = size - 55
            for _ in range(num_to_add):
                ref_atom = pos_target[rng.integers(len(pos_target))]
                for attempt in range(100):
                    offset = rng.normal(0, 0.4, 3)
                    p = ref_atom + offset
                    dists = np.linalg.norm(pos_target - p, axis=1)
                    if np.min(dists) >= 0.5:
                        pos_target = np.vstack([pos_target, p])
                        break
        
        e_seed, pos_seed = relax(pos_target, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Seed 2: Icosahedral symmetry seed
    try:
        pos_seed = _generate_icosahedral_seed(size, rng)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Seed 3: Random compact
    try:
        pos_seed = _generate_random_compact_seed(size, rng)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Seed 4: FCC-based
    try:
        pos_seed = _generate_fcc_seed(size, rng)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Seed 5: Second random compact with different radius
    try:
        rng2 = np.random.default_rng(size * 7919 + 99999)
        pos_seed = _generate_random_compact_seed(size, rng2)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Fallback
    if not seeds:
        radius = 1.0 * size ** (1/3) + 1.0
        points = [rng.uniform(-radius, radius, 3) for _ in range(size)]
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # Relax all seeds and keep the best
    for seed in seeds:
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
    
    if best_positions is None:
        return np.zeros((size, 3))
    
    # Basin Hopping with adaptive step size and multiple restarts
    current_positions = best_positions.copy()
    temperature = 0.3
    step_size = 0.25
    min_step_size = 0.03
    max_step_size = 0.5
    no_improve_count = 0
    max_no_improve = 15
    restart_count = 0
    max_restarts = 3
    
    while time.time() < instance_deadline:
        perturbation = rng.normal(0, step_size, size=(size, 3))
        new_positions = current_positions + perturbation
        
        try:
            new_energy, new_relaxed_positions = relax(new_positions, POTENTIAL)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed_positions.copy()
                current_positions = new_relaxed_positions.copy()
                no_improve_count = 0
                step_size = min(step_size * 1.05, max_step_size)
            else:
                delta_e = new_energy - best_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_positions = new_relaxed_positions.copy()
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Try a restart: new random perturbation from best
                    if restart_count < max_restarts:
                        restart_count += 1
                        big_perturbation = rng.normal(0, 0.8, size=(size, 3))
                        current_positions = best_positions + big_perturbation
                        try:
                            _, current_positions = relax(current_positions, POTENTIAL)
                        except:
                            current_positions = best_positions.copy()
                        step_size = 0.3
                        no_improve_count = 0
                    else:
                        step_size = max(step_size * 0.75, min_step_size)
                        no_improve_count = 0
        
        except Exception:
            continue
        
        temperature *= 0.97
        if temperature < 0.01:
            temperature = 0.01
    
    return best_positions
