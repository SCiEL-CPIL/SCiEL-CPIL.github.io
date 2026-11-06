"""Dzugutov cluster global minima search with icosahedral seeding and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts

def _get_icosahedral_group():
    """
    Generate the 60 rotation matrices of the icosahedral group.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Generate rotations using quaternions or direct matrix construction
    # Simpler: use the fact that the icosahedron has 60 symmetries.
    # We can generate them by combining the basic 72-degree rotation about z-axis
    # and 120-degree rotations about axes to vertices.
    
    # Basic rotation about z-axis by 2*pi/5
    theta = 2 * np.pi / 5
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Rotation about axis (0, 1, phi) normalized
    v = np.array([0, 1, phi])
    v = v / np.linalg.norm(v)
    # Rodrigues rotation formula for 2*pi/3 (120 degrees)
    angle = 2 * np.pi / 3
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    Rv = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Generate group by multiplying these generators
    generators = [Rz, Rv]
    group = [np.eye(3)]
    changed = True
    while changed:
        changed = False
        for g in group:
            for gen in generators:
                new_g = g @ gen
                # Check if new_g is already in group (with tolerance)
                is_new = True
                for existing in group:
                    if np.allclose(new_g, existing, atol=1e-10):
                        is_new = False
                        break
                if is_new:
                    group.append(new_g)
                    changed = True
    return np.array(group)

def _generate_mackay_icosahedron(n_target, spacing=1.0):
    """
    Generate a Mackay icosahedron with exactly n_target atoms if possible (13, 55, 147).
    For other sizes, it generates the closest larger Mackay cluster.
    """
    # Standard Mackay sizes: 13, 55, 147
    if n_target <= 13:
        # Just use icosahedron vertices + center
        ico_verts = _get_icosahedron_vertices()
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts_unit = ico_verts / norms[:, np.newaxis]
        points = [np.zeros(3)] + [v * spacing for v in ico_verts_unit]
        return np.array(points[:n_target])
    elif n_target <= 55:
        # Generate 55 atom cluster
        ico_verts = _get_icosahedron_vertices()
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts_unit = ico_verts / norms[:, np.newaxis]
        
        points = [np.zeros(3)]
        # Shell 1 (12 atoms)
        for v in ico_verts_unit:
            points.append(v * spacing)
            
        # Shell 2 (42 atoms)
        # The 42 atoms in a Mackay icosahedron shell 2 correspond to the icosahedral orbits of a point on the face center or edge midpoint?
        # Actually, the 55-atom cluster is formed by placing an atom at the center of each of the 120 small tetrahedra? No.
        # The 55-atom Mackay icosahedron consists of:
        # 1 center
        # 12 vertices of icosahedron (radius r1)
        # 42 atoms at radius r2. These are the icosahedral orbits of the point (0, 1, phi) scaled?
        # Let's use the group to generate the 42 points.
        # The point (0, 1, phi) is a vertex. Its orbit is 12 points.
        # We need a point that generates an orbit of size 42.
        # The stabilizer of a generic point is trivial (size 1), orbit size 60.
        # The stabilizer of a point on a 2-fold axis (edge midpoint) has size 2, orbit size 30.
        # The stabilizer of a point on a 3-fold axis (face center) has size 3, orbit size 20.
        # The stabilizer of a point on a 5-fold axis (vertex) has size 5, orbit size 12.
        # 12 + 20 + 30 + 60... 
        # Wait, 55 = 1 + 12 + 42.
        # The 42 atoms are composed of two orbits: 20 (face centers) and 22? No.
        # Actually, the 55-atom cluster is often described as having 42 atoms in the second shell.
        # These 42 atoms are the icosahedral orbits of the point corresponding to the "face center" of the icosahedron?
        # Let's calculate the position of the face center of the icosahedron defined by vertices.
        # Vertices: (0, 1, phi), (1, phi, 0), (-1, phi, 0).
        # Midpoint of edge between (0,1,phi) and (1,phi,0) is (0.5, (1+phi)/2, phi/2).
        # Let's just use the known coordinates for the 55-atom Mackay icosahedron.
        # Or better, generate them using the group.
        
        # Let's try the point p = (0, 1, phi) / norm. This is a vertex. Orbit size 12.
        # We need 42 more.
        # The 55-atom cluster is formed by icosahedral shells.
        # Shell 1: 12 atoms at distance d1.
        # Shell 2: 42 atoms at distance d2.
        # The 42 atoms are the icosahedral images of the point (0, 1, phi) * (1/3)? No.
        
        # Let's use a simpler heuristic: The 55-atom cluster can be generated by taking the 12 vertices and adding atoms at the centers of the 20 faces and the midpoints of the 30 edges?
        # 20 + 30 = 50. Plus 12 = 62. Too many.
        # The 42 atoms are actually the icosahedral orbit of the point (1, 1, 1) projected?
        
        # Let's just use the fact that the 55-atom cluster is the union of 5 interpenetrating icosahedra? No.
        
        # I will use a pre-computed set of 42 points that form the second shell of a Mackay icosahedron.
        # These points are the icosahedral orbits of the point (0, 1, phi) scaled by a factor?
        # Actually, the 42 points are the icosahedral orbits of the point (0, 1, phi) * (1/3) + ...?
        
        # Let's use the group to generate an orbit of size 42.
        # This requires a point with stabilizer size 60/42 = 1.42... Impossible.
        # So the 42 atoms must be a union of orbits.
        # 42 = 12 + 30? (Vertices + Edge midpoints)
        # Let's check distances.
        
        # Let's just generate a dense set of points on a sphere and select the 42 that best fit the icosahedral symmetry.
        # Or, simpler: use the known structure.
        # The 55-atom cluster has icosahedral symmetry.
        # I will generate the 12 vertices and 42 points from the icosahedral group acting on a point with a specific stabilizer.
        # If I pick a point on the 2-fold axis (edge midpoint), its orbit is 30.
        # If I pick a point on the 3-fold axis (face center), its orbit is 20.
        # 30 + 20 = 50. Not 42.
        # 12 (vertices) + 30 (edge midpoints) = 42?
        # Let's test this hypothesis.
        # Vertices: 12 points.
        # Edge midpoints: 30 points.
        # Total 42.
        
        # Let's generate edge midpoints.
        # Edges connect vertices at distance d.
        # Vertices are v_i.
        # Midpoint m_ij = (v_i + v_j) / 2.
        
        ico_verts = _get_icosahedron_vertices()
        norms = np.linalg.norm(ico_verts, axis=1)
        ico_verts_unit = ico_verts / norms[:, np.newaxis]
        
        shell2_points = []
        # Add edge midpoints
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts_unit[i] - ico_verts_unit[j])
                # Icosahedron edges have length 2. Face diagonals have length 2*phi.
                # So if d is close to 2.0, it's an edge.
                if abs(d - 2.0) < 0.1:
                    mid = (ico_verts_unit[i] + ico_verts_unit[j]) / 2.0
                    shell2_points.append(mid)
                    
        # We should have 30 midpoints.
        # We need 42. Where are the other 12?
        # Maybe the 42 atoms are NOT just edge midpoints.
        # In a Mackay icosahedron, the second shell has 42 atoms.
        # These are the icosahedral orbits of the point (0, 1, phi) * (1/3)?
        
        # Let's try a different approach: Use the 60 rotation matrices to generate an orbit of a generic point.
        # If I pick a point p, its orbit has size 60 / |Stab(p)|.
        # To get 42, I need |Stab(p)| = 60/42, which is not an integer.
        # So the 42 atoms must be a union of orbits.
        # 42 = 12 + 30.
        # Orbit of vertex (size 12) + Orbit of edge midpoint (size 30).
        # This matches!
        
        # So, Shell 2 consists of:
        # 1. The 12 vertices of the icosahedron (scaled to radius r2)? No, the vertices are Shell 1.
        # Wait, the 55-atom cluster has 1 center, 12 in Shell 1, 42 in Shell 2.
        # If Shell 2 is 12+30, then we have 12 atoms at radius r2a and 30 at r2b?
        # Or are they all at the same radius?
        # In a regular icosahedron, vertices and edge midpoints are at different distances from the center.
        # Distance to vertex: R.
        # Distance to edge midpoint: R * (1/2) * sqrt(1 + cos(2*alpha))?
        # For icosahedron, angle between vertex and edge midpoint is not 90.
        
        # Let's just use the 30 edge midpoints and 12 "face centers"?
        # Face centers orbit size 20.
        # 20 + 30 = 50.
        
        # I will use the 30 edge midpoints and 12 points that are the icosahedral orbit of the face center of the "dual" icosahedron?
        
        # Let's just generate 42 points on a sphere with icosahedral symmetry by taking the icosahedral orbit of a point and adding another orbit if needed.
        # Actually, for the purpose of seeding, we don't need the *exact* Mackay geometry, just a good approximation.
        # I will use the 12 vertices and 30 edge midpoints (total 42) and scale them to a common radius.
        
        shell2_points = np.array(shell2_points)
        # Normalize to unit sphere
        if len(shell2_points) > 0:
            shell2_norms = np.linalg.norm(shell2_points, axis=1)
            shell2_unit = shell2_points / shell2_norms[:, np.newaxis]
        else:
            shell2_unit = np.array([])
            
        r2 = spacing * 1.618 # Golden ratio scaling often used
        for v in shell2_unit:
            points.append(v * r2)
            
        return np.array(points[:n_target])
    else:
        # For n > 55, we can add more shells or just return the 55-atom cluster and let the user handle it?
        # The task asks for 38-60. So we only need up to 60.
        # For 57, 60, we can start with 55 and add atoms.
        return _generate_mackay_icosahedron(55, spacing)

def _generate_defect_icosahedron(n_target, spacing=1.0, rng=None):
    """
    Generate a cluster by starting from a 55-atom Mackay icosahedron
    and removing atoms to reach n_target.
    """
    if rng is None:
        rng = np.random.default_rng(n_target)
    
    base_cluster = _generate_mackay_icosahedron(55, spacing)
    
    if n_target >= 55:
        # Add atoms
        result = base_cluster.copy()
        n_to_add = n_target - 55
        for _ in range(n_to_add):
            idx = rng.integers(0, len(result))
            new_pos = result[idx] + rng.normal(0, spacing * 0.3, 3)
            result = np.vstack([result, new_pos.reshape(1, 3)])
        return result
    
    # Remove atoms
    n_to_remove = 55 - n_target
    # Remove from outer shell (indices 13-54)
    remove_indices = list(range(13, 55))
    rng.shuffle(remove_indices)
    remove_set = set(remove_indices[:n_to_remove])
    
    result = []
    for i, p in enumerate(base_cluster):
        if i not in remove_set:
            result.append(p)
    
    return np.array(result)

def _generate_fcc_seed(n, spacing=1.0, rng=None):
    """Generate an FCC-like seed."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    points = []
    # Generate FCC lattice points
    n_side = int(np.ceil(n ** (1.0/3.0))) + 1
    for i in range(-n_side, n_side+1):
        for j in range(-n_side, n_side+1):
            for k in range(-n_side, n_side+1):
                # FCC basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
                for b in [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]:
                    p = np.array([i, j, k]) + np.array(b)
                    p = p * spacing * 0.5
                    points.append(p)
    
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    max_r = spacing * n ** (1.0/3.0) * 0.8
    mask = norms <= max_r
    points = points[mask]
    
    if len(points) > n:
        idx = rng.choice(len(points), size=n, replace=False)
        points = points[idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                points = np.vstack([points, p.reshape(1, 3)])
    
    return points[:n]


def _local_minimize(positions, potential, parameter):
    """Perform local minimization using the relax function."""
    energy, pos = relax(positions, potential, parameter)
    return energy, pos

def _basin_hopping(positions, potential, parameter, n_trials=20, step_size=0.5, rng=None, time_limit=None):
    """
    Basin hopping with time limit.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = _local_minimize(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    for i in range(n_trials):
        if time_limit and time.time() - start_time > time_limit:
            break
            
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Minimize
        trial_energy, trial_pos = _local_minimize(perturbed, potential, parameter)
        
        # Accept if lower
        if trial_energy < current_energy:
            current_energy = trial_energy
            current_pos = trial_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = trial_pos.copy()
            # Reduce step size if we found a better minimum
            step_size *= 0.9
        else:
            # Increase step size if stuck
            step_size *= 1.1
            step_size = min(step_size, 1.0)
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov clusters using icosahedral seeding and basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin
    
    rng = np.random.default_rng(size)
    
    candidates = []
    
    # 1. Defect-seeded icosahedral structures (main strategy)
    for attempt in range(5):
        if time.time() > deadline:
            break
        pos = _generate_defect_icosahedron(size, spacing=1.0, rng=rng)
        candidates.append(pos)
    
    # 2. FCC-like structures
    for attempt in range(2):
        if time.time() > deadline:
            break
        pos = _generate_fcc_seed(size, spacing=1.0, rng=rng)
        candidates.append(pos)
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_pos = None
    relaxed_candidates = []
    
    for i, pos in enumerate(candidates):
        if time.time() > deadline:
            break
        
        try:
            energy, relaxed_pos = _local_minimize(pos, potential, parameter)
            relaxed_candidates.append((energy, relaxed_pos))
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # Sort relaxed candidates by energy
    relaxed_candidates.sort(key=lambda x: x[0])
    
    # Basin hopping on the top few candidates
    bh_time_limit = 10  # seconds per BH run
    for energy, pos in relaxed_candidates[:3]:
        if time.time() > deadline - 20:
            break
        try:
            bh_energy, bh_pos = _basin_hopping(
                pos, potential, parameter,
                n_trials=30, step_size=0.5, rng=rng,
                time_limit=bh_time_limit
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback
        best_pos = _generate_defect_icosahedron(size, spacing=1.0, rng=rng)
        try:
            _, best_pos = _local_minimize(best_pos, potential, parameter)
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
