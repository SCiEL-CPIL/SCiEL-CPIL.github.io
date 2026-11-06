import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

# Constants
POTENTIAL = 'dzugutov'
MAX_TIME = 300.0

# Global time management
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None
_CALL_COUNT = 0
_GLOBAL_CACHE = {}

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

def _get_mackay_icosahedron_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Structure: 1 center + 12 shell-1 vertices + 42 shell-2 atoms.
    Shell 2: 20 face centers + 12 edge midpoints + 10 additional positions.
    Actually, the standard 55-atom Mackay icosahedron has:
    - 1 center
    - 12 atoms at icosahedron vertices (shell 1)
    - 30 atoms at edge midpoints (shell 2)
    - 12 atoms at face centers? No, that's 1+12+30+12=55.
    Wait, 1+12+30=43, need 12 more. 
    Actually, the 55-atom cluster is 1 + 12 + 42 where 42 = 20 face centers + 12 edge midpoints + 10?
    Let's use a simpler approach: start with 13-atom icosahedron and add 42 atoms
    at positions that maintain icosahedral symmetry as much as possible.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 13-atom icosahedron
    center = np.array([0.0, 0.0, 0.0])
    r1 = 1.1
    atoms_13 = np.vstack([center, verts * r1])
    
    # Get icosahedron faces and edges
    # Faces: 20 triangles. We need to determine which vertices form faces.
    # For a regular icosahedron, each vertex is connected to 5 others.
    # Let's compute faces by checking which triplets of vertices form equilateral triangles
    # with side length equal to the edge length.
    
    # Compute edge length
    v0 = verts[0]
    edge_lengths = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            edge_lengths.append((d, i, j))
    edge_lengths.sort()
    min_edge_len = edge_lengths[0][0]
    
    # Find all edges (pairs with distance close to min_edge_len)
    edges = []
    for d, i, j in edge_lengths:
        if abs(d - min_edge_len) < 1e-6:
            edges.append((i, j))
    
    # Find all faces (triplets of vertices that form triangles with all three edges present)
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edge_set:
                continue
            for k in range(j+1, 12):
                if (i, k) in edge_set and (j, k) in edge_set:
                    faces.append((i, j, k))
    
    # Generate shell 2 atoms:
    # 1. 20 face centers
    face_centers = []
    for face in faces:
        fc = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3
        fc = fc / np.linalg.norm(fc) * (r1 * 1.5)  # Scale to appropriate radius
        face_centers.append(fc)
    
    # 2. 30 edge midpoints
    edge_mids = []
    for edge in edges:
        em = (verts[edge[0]] + verts[edge[1]]) / 2
        em = em / np.linalg.norm(em) * (r1 * 1.4)  # Scale to appropriate radius
        edge_mids.append(em)
    
    # We have 20 + 30 = 50 positions, need 42.
    # Let's take all 20 face centers and 22 of the 30 edge midpoints.
    # Or better: take 20 face centers and 22 edge midpoints that are most symmetric.
    # For simplicity, take all 20 face centers and the first 22 edge midpoints.
    
    shell2_atoms = face_centers + edge_mids[:22]
    
    current_atoms = np.vstack([atoms_13, np.array(shell2_atoms)])
    
    return current_atoms

def _generate_mackay_icosahedron(n_shells=2, radius_scale=1.1):
    """
    Generate a Mackay icosahedron.
    """
    pass

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE, _CALL_COUNT, _GLOBAL_CACHE
    
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
    
    # Check if we have a cached relaxed 55-atom cluster
    if 55 in _GLOBAL_CACHE:
        pos_55_relaxed = _GLOBAL_CACHE[55]
    else:
        # Generate and relax the 55-atom Mackay icosahedron
        try:
            pos_55 = _get_mackay_icosahedron_55()
            e_55, pos_55_relaxed = relax(pos_55, POTENTIAL)
            _GLOBAL_CACHE[55] = pos_55_relaxed
        except Exception:
            pos_55_relaxed = None
    
    # Generate diverse seeds
    seeds = []
    
    # Seed 1: From relaxed 55-atom cluster
    if pos_55_relaxed is not None:
        try:
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
    
    # Seed 2: Random compact with icosahedral bias
    try:
        verts = _get_icosahedron_vertices()
        r_max = 1.2 * size ** (1/3)
        
        points = []
        points.append(np.array([0.0, 0.0, 0.0]))
        
        for _ in range(size - 1):
            placed = False
            for attempt in range(2000):
                if rng.random() < 0.3:
                    v = verts[rng.integers(12)]
                    theta = rng.uniform(-0.5, 0.5)
                    c, s = np.cos(theta), np.sin(theta)
                    v_rot = np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])
                    v_rot = v_rot / np.linalg.norm(v_rot)
                else:
                    v = rng.normal(0, 1, 3)
                    v = v / np.linalg.norm(v)
                
                r = rng.uniform(0.5, r_max)
                p = v * r
                
                if points:
                    dists = np.linalg.norm(np.array(points) - p, axis=1)
                    if np.min(dists) < 0.5:
                        continue
                
                points.append(p)
                placed = True
                break
            
            if not placed:
                p = rng.uniform(-r_max, r_max, 3)
                points.append(p)
        
        pos_seed = np.array(points)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    # Seed 3: Pure random compact
    try:
        radius = 1.0 * size ** (1/3) + 1.0
        points = []
        for _ in range(size):
            placed = False
            for attempt in range(1000):
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
                p = rng.uniform(-radius * 1.2, radius * 1.2, 3)
                points.append(p)
        
        pos_seed = np.array(points)
        e_seed, pos_seed = relax(pos_seed, POTENTIAL)
        seeds.append(pos_seed)
    except Exception:
        pass
    
    if not seeds:
        radius = 1.0 * size ** (1/3) + 1.0
        points = [rng.uniform(-radius, radius, 3) for _ in range(size)]
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
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
    
    # Basin Hopping
    current_positions = best_positions.copy()
    temperature = 0.5
    step_size = 0.3
    min_step_size = 0.05
    no_improve_count = 0
    max_no_improve = 20
    
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
                step_size = min(step_size * 1.1, 0.5)
            else:
                delta_e = new_energy - best_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_positions = new_relaxed_positions.copy()
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    step_size = max(step_size * 0.8, min_step_size)
                    no_improve_count = 0
        
        except Exception:
            continue
        
        temperature *= 0.98
        if temperature < 0.01:
            temperature = 0.01
    
    return best_positions
