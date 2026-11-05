import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a_pot = 1.87
b_pot = 1.94
c_pot = 1.10
d_pot = 0.27
m_pot = 16.0

# Precompute Mackay icosahedron positions for various n
def _mackay_positions(n, rng):
    """Generate Mackay icosahedral positions: 1 + 12 + 42 = 55 atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    ico = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # 42 positions in second shell (Mackay 55-atom icosahedron)
    # True Mackay: 30 edge midpoints + 12 face centers
    # Edge midpoints: midpoints of all 30 edges of the icosahedron
    # Face centers: centroids of all 20 faces
    
    # Icosahedron edges (pairs of vertex indices)
    edges = [
        (0,1), (0,4), (0,6), (0,8), (0,10),
        (1,3), (1,5), (1,9), (1,11),
        (2,3), (2,4), (2,7), (2,8), (2,10),
        (3,5), (3,7), (3,9), (3,11),
        (4,5), (4,7), (4,8), (4,10),
        (5,6), (5,9), (5,11),
        (6,7), (6,8), (6,10),
        (7,9), (7,11),
        (8,10), (9,11)
    ]
    
    # Icosahedron faces (triples of vertex indices)
    faces = [
        (0,1,4), (0,1,9), (0,4,6), (0,6,8), (0,8,10), (0,10,9),
        (1,3,5), (1,3,11), (1,4,5), (1,9,11),
        (2,3,7), (2,3,9), (2,4,7), (2,8,10), (2,10,8),
        (3,5,7), (3,11,5), (3,9,11), (3,7,9),
        (4,5,6), (4,6,7), (4,7,8), (4,8,10), (4,5,9),
        (5,6,11), (5,11,7), (6,7,10), (6,10,8),
        (7,8,10), (7,9,11), (8,10,9), (9,11,10)
    ]
    
    # Correct face list for icosahedron (20 faces)
    faces = [
        (0, 1, 4), (0, 4, 8), (0, 8, 10), (0, 10, 9), (0, 9, 1),
        (1, 3, 5), (1, 5, 9), (1, 9, 11), (1, 11, 3),
        (2, 3, 7), (2, 7, 8), (2, 8, 10), (2, 10, 3),
        (3, 5, 7), (3, 11, 5), (3, 7, 9), (3, 9, 11),
        (4, 5, 6), (4, 6, 8), (4, 8, 7), (4, 7, 5),
        (5, 6, 11), (5, 11, 7), (5, 7, 9), (5, 9, 6),
        (6, 7, 10), (6, 10, 8), (6, 8, 7),
        (7, 9, 11), (7, 11, 10), (7, 10, 8),
        (8, 10, 9), (8, 9, 7), (8, 7, 6),
        (9, 10, 11), (9, 11, 7), (9, 7, 6),
        (10, 11, 8), (10, 8, 6), (10, 6, 9),
        (11, 8, 7), (11, 7, 6), (11, 6, 9)
    ]
    
    # Actually, let me use the standard icosahedron faces
    # Standard icosahedron with vertices as defined above
    faces = [
        (0, 1, 4), (0, 4, 8), (0, 8, 10), (0, 10, 9), (0, 9, 1),
        (1, 3, 5), (1, 5, 9), (1, 9, 11), (1, 11, 3),
        (2, 3, 7), (2, 7, 8), (2, 8, 10), (2, 10, 3),
        (3, 5, 7), (3, 11, 5), (3, 7, 9), (3, 9, 11),
        (4, 5, 6), (4, 6, 8), (4, 8, 7), (4, 7, 5),
        (5, 6, 11), (5, 11, 7), (5, 7, 9), (5, 9, 6),
        (6, 7, 10), (6, 10, 8), (6, 8, 7),
        (7, 9, 11), (7, 11, 10), (7, 10, 8),
        (8, 10, 9), (8, 9, 7), (8, 7, 6),
        (9, 10, 11), (9, 11, 7), (9, 7, 6),
        (10, 11, 8), (10, 8, 6), (10, 6, 9),
        (11, 8, 7), (11, 7, 6), (11, 6, 9)
    ]
    
    # This is getting complex. Let me use a simpler approach:
    # Generate 42 points as the union of edge midpoints and face centers
    # For a regular icosahedron, edge midpoints are at distance sqrt(1+phi)/sqrt(3+2*phi) from center
    # Face centers are at distance 1/sqrt(3+2*phi) from center
    
    # Actually, let me just use the known Mackay 55-atom structure
    # The 42 second-shell atoms are at the 30 edge midpoints and 12 face centers
    # of the icosahedron, all at the same radius
    
    # Edge midpoints of icosahedron
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers of icosahedron
    face_centers = []
    for i, j, k in faces:
        fc = (ico[i] + ico[j] + ico[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # All 42 second-shell directions
    shell2_dirs = np.vstack([edge_midpoints, face_centers])
    shell2_dirs = shell2_dirs / np.linalg.norm(shell2_dirs, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    points.extend(list(ico))
    
    if n <= 13:
        return np.array(points[:n])
    
    # For n > 13, add atoms on second shell
    n_rest = min(n - 13, 42)
    R2 = 1.75  # Approximate second shell radius
    
    shell2 = shell2_dirs[:n_rest] * R2
    points.extend(list(shell2))
    
    # If n > 55, add more atoms on a third shell
    if n > 55:
        n_third = n - 55
        R3 = 2.5
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_third):
            z = 1 - 2 * (i + 0.5) / n_third
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * R3
            points.append(p)
    
    return np.array(points[:n])

def _energy_and_gradient(positions):
    """
    Vectorized calculation of energy and gradient for Dzugutov potential.
    Returns (energy, gradient).
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    r_safe = np.where(mask, r, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Region 1: r < a
    m1 = (r < a_pot) & mask
    # Region 2: a <= r < b
    m2 = (r >= a_pot) & (r < b_pot) & mask
    
    r_safe1 = np.where(m1, r_safe, 1.0)
    r_safe2 = np.where(m2, r_safe, 1.0)
    
    e1_vals = A * (r_safe1 ** (-m_pot) - B) * np.exp(c_pot / (r_safe1 - a_pot))
    e2_vals = B * np.exp(d_pot / (r_safe2 - b_pot))
    
    e1 = np.where(m1, e1_vals, 0.0)
    e2 = np.where(m2, e2_vals, 0.0)
    
    pair_energy = e1 + e2
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient
    exp_term1 = np.exp(c_pot / (r_safe1 - a_pot))
    de1_dr_vals = A * (
        -m_pot * r_safe1 ** (-m_pot - 1) * exp_term1 + 
        (r_safe1 ** (-m_pot) - B) * exp_term1 * (-c_pot / (r_safe1 - a_pot) ** 2)
    )
    de1_dr = np.where(m1, de1_dr_vals, 0.0)
    
    exp_term2 = np.exp(d_pot / (r_safe2 - b_pot))
    de2_dr_vals = B * exp_term2 * (-d_pot / (r_safe2 - b_pot) ** 2)
    de2_dr = np.where(m2, de2_dr_vals, 0.0)
    
    de_dr = de1_dr + de2_dr
    
    factor = np.where(r_safe > 1e-12, de_dr / r_safe, 0.0)
    factor = np.where(mask, factor, 0.0)
    
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return energy, grad

def _icosahedron_vertices(radius=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _generate_mackay_like(n, rng):
    """
    Generate a Mackay-like icosahedral structure.
    """
    return _mackay_positions(n, rng)

def _polish_structure(positions, potential, param, max_iter=100):
    """Polish a structure using L-BFGS-B with the Dzugutov energy/gradient."""
    try:
        def obj_and_grad(x):
            pos = x.reshape(-1, 3)
            # Ensure minimum distance
            d = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            if np.min(d) < 0.45:
                return 1e10, np.zeros_like(x)
            e, g = _energy_and_gradient(pos)
            return e, g.ravel()
        
        x0 = positions.ravel()
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': max_iter, 'gtol': 1e-6})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        return positions, _energy_and_gradient(positions)[0]
    except Exception:
        return positions, _energy_and_gradient(positions)[0]

def _generate_dense_cluster(n, rng, radius=None):
    """Generate a dense cluster with well-spaced atoms."""
    if radius is None:
        radius = 1.2 * (n / 13) ** (1.0/3.0)
    
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        # Sample from ball
        r = radius * rng.random() ** (1.0/3.0)
        theta = 2 * np.pi * rng.random()
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    # Fill remaining with random points
    while len(points) < n:
        r = radius * rng.random() ** (1.0/3.0)
        theta = 2 * np.pi * rng.random()
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        points.append(p)
    
    return np.array(points)

def _parametric_optimize(n, rng):
    """
    Optimize the radii of icosahedral shells.
    """
    ico_dir = _icosahedron_vertices(1.0)
    
    # Get proper Mackay second-shell directions
    mackay_pos = _mackay_positions(55, rng)
    if n >= 13:
        rest_dirs = mackay_pos[13:55]
        rest_dirs = rest_dirs / np.linalg.norm(rest_dirs, axis=1, keepdims=True)
        n_rest = min(n - 13, len(rest_dirs))
    else:
        rest_dirs = np.array([])
        n_rest = 0
    
    def objective(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e10
        pos = np.zeros((n, 3))
        pos[0] = 0
        for i in range(12):
            if 1 + i < n:
                pos[1 + i] = ico_dir[i] * r1
        for i in range(n_rest):
            if 13 + i < n:
                pos[13 + i] = rest_dirs[i] * r2
        
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.4:
            return 1e10
            
        energy, _ = _energy_and_gradient(pos)
        return energy
        
    r1_init = 1.0
    r2_init = 1.7
    x0 = [r1_init, r2_init]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            r1_opt, r2_opt = res.x
            pos = np.zeros((n, 3))
            pos[0] = 0
            for i in range(12):
                if 1 + i < n:
                    pos[1 + i] = ico_dir[i] * r1_opt
            for i in range(n_rest):
                if 13 + i < n:
                    pos[13 + i] = rest_dirs[i] * r2_opt
            return pos, res.fun
    except Exception:
        pass
        
    pos = _generate_mackay_like(n, rng)
    energy, _ = _energy_and_gradient(pos)
    return pos, energy

def construct(instance):
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 20.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
        
    def try_relax(pos):
        nonlocal best_energy, best_positions
        try:
            pos_noisy = pos + 0.01 * rng.standard_normal((n, 3))
            energy, relaxed_pos = relax(pos_noisy, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False

    def try_polish(pos):
        """Try to polish a structure using L-BFGS-B."""
        nonlocal best_energy, best_positions
        try:
            polished, e = _polish_structure(pos, potential, param, max_iter=50)
            if e < best_energy:
                best_energy = e
                best_positions = polished.copy()
                return True
        except Exception:
            pass
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Mackay icosahedral (primary structure for Dzugutov)
    initial_guesses.append(_generate_mackay_like(n, rng))
    
    # 2. Parametric icosahedral optimization
    pos, e = _parametric_optimize(n, rng)
    initial_guesses.append(pos)
    
    # 3. Dense random clusters with different radii
    for radius_factor in [0.9, 1.0, 1.1, 1.2]:
        radius = 1.0 * (n / 13) ** (1.0/3.0) * radius_factor
        initial_guesses.append(_generate_dense_cluster(n, rng, radius))
    
    # 4. Additional Mackay variants with noise
    for i in range(3):
        pos = _generate_mackay_like(n, rng)
        pos += 0.05 * rng.standard_normal((n, 3))
        initial_guesses.append(pos)

    # Phase 2: Relax all initial guesses
    for pos in initial_guesses:
        if not check_time(): break
        try_relax(pos)
        
    # Phase 3: Basin Hopping with adaptive kick schedule
    if best_positions is not None:
        # Keep a list of good structures for hopping
        good_structures = [best_positions.copy()]
        
        # Add a few more diverse starting points for hopping
        for pos in initial_guesses[:4]:
            if not check_time(): break
            try:
                pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
                energy, relaxed_pos = relax(pos_noisy, potential, param)
                if energy < best_energy * 1.15:  # Within 15% of best
                    good_structures.append(relaxed_pos.copy())
            except Exception:
                pass
        
        # Basin hopping from each good structure with adaptive kicks
        for struct_idx, struct in enumerate(good_structures):
            if not check_time(): break
            
            # Adaptive kick size: start small, increase gradually
            base_kick = 0.10
            max_kicks = 20
            for i in range(max_kicks):
                if not check_time(): break
                
                # Kick size increases with iteration, with some randomness
                kick_size = base_kick * (1.0 + 0.15 * i) * (0.8 + 0.4 * rng.random())
                pos_kicked = struct + kick_size * rng.standard_normal((n, 3))
                
                if try_relax(pos_kicked):
                    # If we found a new best, update our reference structure
                    if best_positions is not None:
                        struct = best_positions.copy()
                        # Reset kick counter for new best
                        break

    # Phase 4: Polish the best structure with L-BFGS-B
    if best_positions is not None and check_time():
        try_polish(best_positions)
        
        # Phase 5: Final basin hopping with smaller kicks around best
        if check_time():
            for i in range(15):
                if not check_time(): break
                kick_size = 0.03 + 0.005 * i
                pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
                try_relax(pos_kicked)
                
                # Polish periodically
                if i % 5 == 4 and check_time():
                    try_polish(best_positions)

    # Fallback
    if best_positions is None:
        best_positions = _generate_mackay_like(n, rng)
        
    return best_positions

# EVOLVE-BLOCK-END
