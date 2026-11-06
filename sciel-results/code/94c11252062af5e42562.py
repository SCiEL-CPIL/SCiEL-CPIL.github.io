import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0
_LOCK = threading.Lock()
_best_energy = float('inf')
_best_pos = None

def _time_left():
    return _DEADLINE - time.time()

def _get_icosahedron_verts():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _build_polytetrahedral(n, seed=0):
    """
    Build a polytetrahedral cluster by assembling icosahedral units.
    Each unit is an icosahedron (13 atoms: 1 center + 12 shell).
    Units share up to 12 shell atoms (face-sharing) or fewer (edge/vertex-sharing).
    We represent the structure as a list of icosahedral units with their
    relative orientations (quaternions) and positions.
    
    For n atoms, we use approximately n/13 units (with sharing).
    """
    rng = np.random.default_rng(seed)
    ico_verts = _get_icosahedron_verts()  # 12 unit vectors
    ico_full = np.vstack([np.zeros(3), ico_verts])  # 13 atoms: center + 12
    
    # Determine number of icosahedral units
    # In a polytetrahedral structure, adjacent icosahedra share a face (3 atoms)
    # or edge (2 atoms) or vertex (1 atom)
    # For simplicity, we'll use a chain/ring of icosahedra sharing vertices
    n_units = max(1, n // 13)
    remainder = n - n_units * 13
    
    # Start with first icosahedron at origin
    units = []
    current_pos = np.zeros(3)
    current_rot = np.eye(3)
    
    for i in range(n_units):
        units.append({'pos': current_pos.copy(), 'rot': current_rot.copy()})
        
        if i < n_units - 1:
            # Choose a random face of current icosahedron to share with next
            # A face is defined by 3 vertices
            face_indices = rng.choice(12, size=3, replace=False)
            face_center = ico_verts[face_indices].mean(axis=0)
            face_center_dir = face_center / np.linalg.norm(face_center)
            
            # New icosahedron center is displaced along face normal
            # Distance: two icosahedra sharing a face have centers separated by
            # approximately 2 * r_ico * cos(angle) where angle relates to face geometry
            # For regular icosahedron with circumradius 1, face inradius ~ 0.795
            # Center-to-face-center distance ~ 0.795
            # When two icosahedra share a face, centers are ~ 2*0.795 = 1.59 apart
            # But we need to ensure atoms are >= 0.5 apart, so let's use ~1.6
            displacement = 1.6 * face_center_dir
            
            # Random rotation for the new icosahedron
            theta = rng.uniform(0, 2 * np.pi)
            rot_axis = rng.normal(size=3)
            rot_axis /= np.linalg.norm(rot_axis)
            # Simple rotation matrix from axis-angle
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                          [rot_axis[2], 0, -rot_axis[0]],
                          [-rot_axis[1], rot_axis[0], 0]])
            K2 = K @ K
            new_rot = current_rot @ (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K2)
            
            current_pos = current_pos + displacement
            current_rot = new_rot
    
    # Assemble atoms
    atoms = []
    used_atoms = set()
    
    for unit in units:
        pos = unit['pos']
        rot = unit['rot']
        
        for j, v in enumerate(ico_full):
            world_pos = pos + rot @ v
            # Round to avoid floating point issues in deduplication
            key = tuple(np.round(world_pos, 4))
            if key not in used_atoms:
                used_atoms.add(key)
                atoms.append(world_pos)
    
    # If we have more atoms than needed, trim
    if len(atoms) > n:
        atoms = atoms[:n]
    # If we have fewer, add random atoms near existing ones
    while len(atoms) < n:
        base = atoms[rng.randint(len(atoms))]
        offset = rng.normal(0, 0.3, 3)
        new_atom = base + offset
        # Check minimum distance
        if all(np.linalg.norm(new_atom - a) >= 0.5 for a in atoms):
            atoms.append(new_atom)
        else:
            atoms.append(new_atom)  # Add anyway, relax will fix
    
    return np.array(atoms)

def _random_seed(n, scale, seed):
    rng = np.random.default_rng(seed)
    radius = scale
    points = []
    min_dist = 0.55
    max_attempts = n * 200
    for _ in range(max_attempts):
        if len(points) >= n: break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) < radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _relax_and_evaluate(pos, n, potential, parameter):
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        if relaxed_pos is not None and len(relaxed_pos) == n:
            min_pair_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
            if min_pair_dist >= 0.5:
                return energy, relaxed_pos
        return energy, pos
    except Exception:
        return float('inf'), pos

def _hinge_perturb(pos, n, seed):
    """
    Perturb the structure by rotating groups of atoms around shared axes.
    This preserves local icosahedral order while allowing global rearrangement.
    """
    rng = np.random.default_rng(seed)
    new_pos = pos.copy()
    
    # Identify clusters of nearby atoms (local icosahedral units)
    # Simple approach: pick a random atom, find its 12 nearest neighbors,
    # rotate that group around a random axis through the center atom
    if n < 13:
        # Small cluster: just do random perturbation
        return pos + rng.normal(0, 0.2, pos.shape)
    
    # Pick a random "hinge" atom
    hinge_idx = rng.randint(n)
    hinge_pos = pos[hinge_idx]
    
    # Find the 12 nearest neighbors of the hinge atom
    dists = np.array([np.linalg.norm(pos[i] - hinge_pos) for i in range(n) if i != hinge_idx])
    neighbor_indices = [i for i in range(n) if i != hinge_idx]
    neighbor_indices.sort(key=lambda i: np.linalg.norm(pos[i] - hinge_pos))
    group = [hinge_idx] + neighbor_indices[:12]
    
    # Random rotation axis through hinge
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-0.5, 0.5)
    
    # Apply rotation to the group
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    K2 = K @ K
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K2
    
    for idx in group:
        if idx == hinge_idx:
            continue
        rel = pos[idx] - hinge_pos
        new_pos[idx] = hinge_pos + R @ rel
    
    return new_pos

def _basin_hopping_hinge(pos, n, potential, parameter, n_steps=25, seed=0):
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_and_evaluate(pos, n, potential, parameter)
    local_best_energy = current_energy
    local_best_pos = current_pos.copy()
    
    for i in range(n_steps):
        if _time_left() < 8.0:
            break
        
        # Hinge perturbation: preserve local order
        perturbed = _hinge_perturb(current_pos, n, seed + i)
        new_energy, new_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Metropolis criterion
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < local_best_energy:
                local_best_energy = new_energy
                local_best_pos = new_pos.copy()
        else:
            delta = new_energy - current_energy
            temp = 0.3 * (0.97 ** i)
            accept_prob = np.exp(-delta / temp)
            if rng.random() < accept_prob:
                current_energy = new_energy
                current_pos = new_pos
    
    with _LOCK:
        if local_best_energy < _best_energy:
            _best_energy = local_best_energy
            _best_pos = local_best_pos.copy()
            
    return local_best_energy, local_best_pos

def _genetic_evolve(n, potential, parameter, pop_size=8, n_gen=10, seed=0):
    """
    Genetic algorithm over polytetrahedral structures.
    Genomes: list of hinge angles and unit positions.
    We encode the structure as the initial polytetrahedral build parameters
    and evolve them.
    """
    global _best_energy, _best_pos
    
    rng = np.random.default_rng(seed)
    
    # Initialize population with diverse polytetrahedral structures
    population = []
    for i in range(pop_size):
        pos = _build_polytetrahedral(n, seed=seed + i)
        energy, relaxed = _relax_and_evaluate(pos, n, potential, parameter)
        population.append((energy, relaxed))
    
    # Sort by energy
    population.sort(key=lambda x: x[0])
    
    best_energy, best_pos = population[0]
    
    for gen in range(n_gen):
        if _time_left() < 15.0:
            break
        
        # Create next generation
        next_gen = []
        
        # Elitism: keep top half
        for i in range(pop_size // 2):
            next_gen.append(population[i])
        
        # Crossover and mutation for the rest
        while len(next_gen) < pop_size:
            # Pick parents (tournament selection)
            p1_idx = rng.choice(pop_size // 2)
            p2_idx = rng.choice(pop_size // 2)
            p1 = population[p1_idx][1]
            p2 = population[p2_idx][1]
            
            # Crossover: mix positions
            child = (p1 + p2) / 2.0
            
            # Mutation: hinge perturbation
            if rng.random() < 0.7:
                child = _hinge_perturb(child, n, seed + gen * 100 + len(next_gen))
            else:
                child = child + rng.normal(0, 0.15, child.shape)
            
            energy, relaxed = _relax_and_evaluate(child, n, potential, parameter)
            next_gen.append((energy, relaxed))
        
        population = next_gen
        population.sort(key=lambda x: x[0])
        
        if population[0][0] < best_energy:
            best_energy = population[0][0]
            best_pos = population[0][1].copy()
    
    with _LOCK:
        if best_energy < _best_energy:
            _best_energy = best_energy
            _best_pos = best_pos.copy()
    
    return best_energy, best_pos

def construct(instance):
    global _best_energy, _best_pos
    _best_energy = float('inf')
    _best_pos = None
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds using polytetrahedral construction
    pt_seeds = []
    for i in range(6):
        pt_seeds.append(_build_polytetrahedral(n, seed=100 + i))
    
    # Add a couple random seeds for diversity
    est_r = (3 * n / (4 * np.pi)) ** (1/3) * 0.75
    pt_seeds.append(_random_seed(n, est_r, 42))
    pt_seeds.append(_random_seed(n, est_r * 1.1, 43))
    
    # Phase 1: Parallel basin hopping with hinge perturbations (4 threads)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx in range(4):
            seed_pos = pt_seeds[idx]
            future = executor.submit(_basin_hopping_hinge, seed_pos, n, potential, 
                                     parameter, n_steps=30, seed=200 + idx)
            futures.append(future)
        
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
    
    # Phase 2: Genetic evolution on remaining time
    if _time_left() > 30.0:
        _genetic_evolve(n, potential, parameter, pop_size=6, n_gen=8, seed=500)
    
    # Phase 3: Final refinement
    if _best_pos is not None and _time_left() > 15.0:
        _basin_hopping_hinge(_best_pos, n, potential, parameter, 
                             n_steps=15, seed=999)
    
    if _best_pos is None:
        _best_pos = pt_seeds[0]
        
    return _best_pos
