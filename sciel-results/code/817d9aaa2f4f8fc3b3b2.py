import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize
from itertools import combinations

# Dzugutov potential parameters
A = 5.82
B = 1.28
a_pot = 1.87
b_pot = 1.94
c_pot = 1.10
d_pot = 0.27
m_pot = 16.0

# Cache for Mackay 55 positions
_MACKAY_55_CACHE = None

def _get_mackay_55_positions():
    """
    Precompute the 55 Mackay icosahedron positions (1 + 12 + 42).
    Returns a (55, 3) array.
    """
    global _MACKAY_55_CACHE
    if _MACKAY_55_CACHE is not None:
        return _MACKAY_55_CACHE.copy()
    
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
    
    n_ico = len(ico)
    edges = []
    faces = []
    
    # Find edges: pairs at minimum distance
    dists = np.zeros((n_ico, n_ico))
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            dists[i,j] = np.linalg.norm(ico[i] - ico[j])
            dists[j,i] = dists[i,j]
    
    min_dist_ico = np.min(dists[np.triu_indices(n_ico, k=1)])
    
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            if abs(dists[i,j] - min_dist_ico) < 1e-6:
                edges.append((i, j))
    
    # Find faces: triples of vertices where all three pairs are edges
    edge_set = set()
    for e in edges:
        edge_set.add(tuple(sorted(e)))
    
    for i, j, k in combinations(range(n_ico), 3):
        if (min(i,j), max(i,j)) in edge_set and (min(i,k), max(i,k)) in edge_set and (min(j,k), max(j,k)) in edge_set:
            faces.append((i, j, k))
    
    # Compute face centers and edge midpoints
    second_shell = []
    
    # 20 face centers
    for face in faces:
        center = (ico[face[0]] + ico[face[1]] + ico[face[2]]) / 3.0
        center = center / np.linalg.norm(center)
        second_shell.append(center)
    
    # 30 edge midpoints
    for edge in edges:
        mid = (ico[edge[0]] + ico[edge[1]]) / 2.0
        mid = mid / np.linalg.norm(mid)
        second_shell.append(mid)
    
    second_shell = np.array(second_shell)
    
    points = [np.zeros(3)]
    points.extend(list(ico))
    points.extend(list(second_shell))
    
    result = np.array(points)
    _MACKAY_55_CACHE = result
    return result.copy()

def _find_r_min():
    """Find the minimum of the Dzugutov potential."""
    from scipy.optimize import minimize_scalar
    def V(r):
        if r < a_pot:
            return A * (r**(-m_pot) - B) * np.exp(c_pot / (r - a_pot))
        elif r < b_pot:
            return B * np.exp(d_pot / (r - b_pot))
        else:
            return 0.0
    res = minimize_scalar(V, bounds=(0.5, 2.5), method='bounded', options={'xatol': 1e-8})
    return res.x

_R_MIN = None
def get_r_min():
    global _R_MIN
    if _R_MIN is None:
        _R_MIN = _find_r_min()
    return _R_MIN

def _mackay_positions(n, rng):
    """Generate Mackay icosahedral positions for n atoms."""
    mackay55 = _get_mackay_55_positions()
    
    if n <= 55:
        return mackay55[:n].copy()
    else:
        # For n > 55, add extra atoms on a third shell
        n_extra = n - 55
        golden_angle = np.pi * (3 - np.sqrt(5))
        R3 = 2.5
        extra_points = []
        for i in range(n_extra):
            z = 1 - 2 * (i + 0.5) / n_extra
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * R3
            extra_points.append(p)
        
        points = np.vstack([mackay55, np.array(extra_points)])
        return points[:n].copy()

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
    r = np.where(mask, r, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Region 1: r < a
    m1 = r < a_pot
    # Region 2: a <= r < b
    m2 = (r >= a_pot) & (r < b_pot)
    
    r_safe1 = np.where(m1, r, 1.0)
    r_safe2 = np.where(m2, r, 1.0)
    
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
    
    factor = np.where(r > 1e-12, de_dr / r, 0.0)
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
    """Generate a Mackay-like icosahedral structure."""
    pos = _mackay_positions(n, rng)
    return pos

def _generate_dense_cluster(n, rng, radius=None):
    """Generate a dense cluster with well-spaced atoms."""
    if radius is None:
        radius = 1.2 * (n / 13) ** (1.0/3.0)
    
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
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

def _polish_structure(pos, potential, param, time_budget=5.0):
    """
    Polish a structure using scipy L-BFGS-B with the Dzugutov energy/gradient.
    Returns (polished_pos, energy).
    """
    n = len(pos)
    pos_flat = pos.flatten().copy()
    
    def obj_and_grad(x):
        p = x.reshape(n, 3)
        if n > 1:
            d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            if np.min(d) < 0.49:
                return 1e10, np.zeros_like(x)
        e, g = _energy_and_gradient(p)
        return e, g.flatten()
    
    try:
        start_t = time.time()
        res = minimize(obj_and_grad, pos_flat, method='L-BFGS-B', 
                       jac=True, options={'maxiter': 300, 'gtol': 1e-7, 'maxfun': 1000})
        elapsed = time.time() - start_t
        
        if res.fun < 1e9:
            polished = res.x.reshape(n, 3)
            if n > 1:
                d = np.linalg.norm(polished[:, np.newaxis, :] - polished[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                if np.min(d) >= 0.5 - 1e-6:
                    return polished, res.fun
    except Exception:
        pass
    
    return pos, _energy_and_gradient(pos)[0]

def _parametric_optimize(n, rng):
    """
    Optimize the radii of icosahedral shells using proper Mackay geometry.
    """
    ico_dir = _icosahedron_vertices(1.0)
    mackay_pos = _mackay_positions(55, rng)
    n_ico = 13
    n_second = 42
    
    if n <= 13:
        second_dirs = np.array([])
    elif n <= 55:
        second_dirs = mackay_pos[n_ico:n]
        second_dirs = second_dirs / np.linalg.norm(second_dirs, axis=1, keepdims=True)
    else:
        second_dirs = mackay_pos[n_ico:n_ico+n_second]
        second_dirs = second_dirs / np.linalg.norm(second_dirs, axis=1, keepdims=True)
        n_extra = n - 55
        golden_angle = np.pi * (3 - np.sqrt(5))
        extra_dirs = []
        for i in range(n_extra):
            z = 1 - 2 * (i + 0.5) / n_extra
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            d = np.array([r * np.cos(theta), r * np.sin(theta), z])
            extra_dirs.append(d)
        if extra_dirs:
            second_dirs = np.vstack([second_dirs, np.array(extra_dirs)])
    
    n_rest = len(second_dirs)
    
    def objective(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e10
        pos = np.zeros((n, 3))
        pos[0] = 0
        for i in range(min(12, n-1)):
            pos[1 + i] = ico_dir[i] * r1
        for i in range(min(n_rest, n-13)):
            if 13 + i < n:
                pos[13 + i] = second_dirs[i] * r2
        
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
            for i in range(min(12, n-1)):
                pos[1 + i] = ico_dir[i] * r1_opt
            for i in range(min(n_rest, n-13)):
                if 13 + i < n:
                    pos[13 + i] = second_dirs[i] * r2_opt
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

    def try_relax_direct(pos):
        nonlocal best_energy, best_positions
        try:
            energy, relaxed_pos = relax(pos, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False

    def try_polish(pos):
        nonlocal best_energy, best_positions
        if not check_time():
            return False
        try:
            polished, e = _polish_structure(pos, potential, param, time_budget=4.0)
            if e < best_energy:
                energy, relaxed_pos = relax(polished, potential, param)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
                    return True
        except Exception:
            pass
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Mackay icosahedral
    initial_guesses.append(_generate_mackay_like(n, rng))
    
    # 2. Parametric icosahedral optimization
    pos, e = _parametric_optimize(n, rng)
    initial_guesses.append(pos)
    
    # 3. Dense random clusters
    for radius_factor in [0.85, 0.95, 1.05, 1.15]:
        radius = 1.0 * (n / 13) ** (1.0/3.0) * radius_factor
        initial_guesses.append(_generate_dense_cluster(n, rng, radius))
    
    # 4. Additional Mackay variants with noise
    for noise_level in [0.05, 0.1, 0.15]:
        pos = _generate_mackay_like(n, rng)
        pos += noise_level * rng.standard_normal((n, 3))
        initial_guesses.append(pos)
    
    # 5. Scaled Mackay structures
    for scale in [0.95, 1.05]:
        pos = _generate_mackay_like(n, rng) * scale
        initial_guesses.append(pos)

    # Phase 2: Relax all initial guesses
    for pos in initial_guesses:
        if not check_time(): break
        try_relax(pos)
        
    # Phase 2.5: Polish the best structure found so far
    if best_positions is not None and check_time():
        try_polish(best_positions)
        
    # Phase 3: Basin Hopping from multiple good structures
    if best_positions is not None:
        good_structures = [best_positions.copy()]
        
        for pos in initial_guesses[:5]:
            if not check_time(): break
            try:
                pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
                energy, relaxed_pos = relax(pos_noisy, potential, param)
                if energy < best_energy * 1.1:
                    good_structures.append(relaxed_pos.copy())
            except Exception:
                pass
        
        # Basin hopping with adaptive kick schedule
        for struct_idx, struct in enumerate(good_structures):
            if not check_time(): break
            
            n_hops = 30
            for i in range(n_hops):
                if not check_time(): break
                
                if i < n_hops // 3:
                    kick_size = 0.1 + 0.05 * (i / (n_hops // 3))
                elif i < 2 * n_hops // 3:
                    kick_size = 0.25
                else:
                    kick_size = 0.15 - 0.05 * ((i - 2 * n_hops // 3) / (n_hops // 3))
                
                pos_kicked = struct + kick_size * rng.standard_normal((n, 3))
                try_relax(pos_kicked)
                
                if best_positions is not None and np.linalg.norm(best_positions - struct) > 0.3:
                    struct = best_positions.copy()
                    break

    # Phase 4: Final polish with smaller kicks
    if best_positions is not None and check_time():
        for i in range(25):
            if not check_time(): break
            kick_size = 0.02 + 0.01 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked)
        
        if check_time():
            try_polish(best_positions)
        
        if check_time():
            try_relax_direct(best_positions)

    # Fallback
    if best_positions is None:
        best_positions = _generate_mackay_like(n, rng)
        
    return best_positions
