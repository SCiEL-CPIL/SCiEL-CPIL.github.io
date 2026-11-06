"""Combined program: multi-polyhedral seeds + basin hopping + L-BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 40s safety margin for 330s limit


def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT


def _get_icosahedral_points():
    """Returns 12 vertices of a regular icosahedron, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norm = np.linalg.norm(v[0])
    v = v / norm
    return v


def _get_dodecahedral_points():
    """Returns 20 vertices of a regular dodecahedron, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / phi
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi]
    ], dtype=float)
    norm = np.linalg.norm(v[0])
    v = v / norm
    return v


def _get_cuboctahedral_points():
    """Returns 12 vertices of a cuboctahedron, normalized to unit radius."""
    v = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    norm = np.linalg.norm(v[0])
    v = v / norm
    return v


def _get_face_centers(vertices):
    """Compute face centers from a set of vertices using Delaunay-like approach."""
    n = len(vertices)
    dists = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(dists[i, j], min_dist, rtol=1e-6):
                edges.append((i, j))
    
    face_centers = []
    seen = set()
    for i, j in edges:
        for k in range(n):
            if k == i or k == j:
                continue
            if np.isclose(dists[i, k], min_dist, rtol=1e-6) and np.isclose(dists[j, k], min_dist, rtol=1e-6):
                key = tuple(sorted([i, j, k]))
                if key not in seen:
                    seen.add(key)
                    center = (vertices[i] + vertices[j] + vertices[k]) / 3.0
                    center = center / np.linalg.norm(center)
                    face_centers.append(center)
    return np.array(face_centers) if face_centers else np.array([])


def _get_edge_centers(vertices):
    """Compute edge midpoints from a set of vertices."""
    n = len(vertices)
    dists = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    edge_centers = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(dists[i, j], min_dist, rtol=1e-6):
                center = (vertices[i] + vertices[j]) / 2.0
                center = center / np.linalg.norm(center)
                edge_centers.append(center)
    return np.array(edge_centers) if edge_centers else np.array([])


# Precompute polyhedral point sets
_ICO_VERT = _get_icosahedral_points()
_ICO_FACE = _get_face_centers(_ICO_VERT)
_ICO_EDGE = _get_edge_centers(_ICO_VERT)

_DODE_VERT = _get_dodecahedral_points()
_DODE_FACE = _get_face_centers(_DODE_VERT)
_DODE_EDGE = _get_edge_centers(_DODE_VERT)

_CUBO_VERT = _get_cuboctahedral_points()
_CUBO_FACE = _get_face_centers(_CUBO_VERT)
_CUBO_EDGE = _get_edge_centers(_CUBO_VERT)


def _generate_seed(n, rng, poly_type, scale=1.0):
    """
    Generate n points based on polyhedral symmetry.
    poly_type: 'ico', 'dode', 'cubo', 'random'
    """
    if poly_type == 'random':
        radius = 1.3 * (n / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                ok = True
                for q in pts:
                    if np.linalg.norm(p - np.array(q)) < 0.52:
                        ok = False
                        break
                if ok:
                    pts.append(p.tolist())
            attempts += 1
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
        return np.array(pts)
    
    # Polyhedral types: build concentric shells
    if poly_type == 'ico':
        verts = _ICO_VERT
        faces = _ICO_FACE
        edges = _ICO_EDGE
    elif poly_type == 'dode':
        verts = _DODE_VERT
        faces = _DODE_FACE
        edges = _DODE_EDGE
    elif poly_type == 'cubo':
        verts = _CUBO_VERT
        faces = _CUBO_FACE
        edges = _CUBO_EDGE
    else:
        verts = _ICO_VERT
        faces = _ICO_FACE
        edges = _ICO_EDGE
    
    r1 = 1.05 * scale
    points = [np.array([0.0, 0.0, 0.0])]
    shell1 = verts * r1
    points.extend(shell1.tolist())
    
    if n <= len(points):
        return np.array(points[:n])
    
    needed = n - len(points)
    
    # Build pool of outer shell points
    outer_pool = []
    if len(faces) > 0:
        outer_pool.extend(faces.tolist())
    if len(edges) > 0:
        outer_pool.extend(edges.tolist())
    
    if len(outer_pool) == 0:
        outer_pool = verts.tolist()
    
    # If we need more than available in structured pool, add random points
    r2 = r1 * 1.55
    rng_idx = rng.permutation(len(outer_pool))
    selected = np.array(outer_pool)[rng_idx[:min(needed, len(outer_pool))]] * r2
    points.extend(selected.tolist())
    
    # Fill remaining with random points on a shell
    remaining = n - len(points)
    if remaining > 0:
        for _ in range(remaining):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            p = direction * r2 * rng.uniform(0.9, 1.1)
            points.append(p.tolist())
    
    return np.array(points[:n])


def _l_bfgs_refine(positions, potential, parameter, max_iter=300):
    """
    Use L-BFGS-B from scipy for efficient local minimization.
    """
    from scipy.optimize import minimize
    
    pos = positions.copy()
    
    try:
        energy, grad = dzugutov_energy_gradient(pos)
    except Exception:
        return np.inf, positions
    
    if not np.isfinite(energy):
        return np.inf, positions
    
    def objective_flat(flat_pos):
        p = flat_pos.reshape(-1, 3)
        try:
            e, g = dzugutov_energy_gradient(p)
            if not np.isfinite(e):
                return 1e10, np.zeros_like(flat_pos)
            return e, g.ravel()
        except Exception:
            return 1e10, np.zeros_like(flat_pos)
    
    try:
        result = minimize(
            objective_flat,
            pos.ravel(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-10}
        )
        if np.isfinite(result.fun):
            best_pos = result.x.reshape(-1, 3)
            best_energy = result.fun
            return best_energy, best_pos
    except Exception:
        pass
    
    return energy, pos


def _basin_hopping(positions, potential, parameter, rng, n_cycles=50, kick_scale=0.15):
    """
    Basin hopping: alternate between random kicks and local minimization.
    Uses Metropolis acceptance based on energy difference.
    """
    pos = positions.copy()
    try:
        energy, _ = dzugutov_energy_gradient(pos)
    except Exception:
        return np.inf, positions
    
    if not np.isfinite(energy):
        return np.inf, positions
    
    best_energy = energy
    best_pos = pos.copy()
    
    t = 1.0
    for i in range(n_cycles):
        if not _check_time():
            break
        
        # Random kick
        kick = rng.normal(0, kick_scale, size=pos.shape)
        new_pos = pos + kick
        
        # Check minimum distance
        dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue
        
        # Local minimization
        new_energy, new_relaxed = _l_bfgs_refine(new_pos, potential, parameter, max_iter=150)
        
        if not np.isfinite(new_energy):
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
            pos = new_relaxed.copy()
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        # Cool down
        t *= 0.95
    
    return best_energy, best_pos


def _kick_and_relax(positions, potential, parameter, rng, n_kicks=10, kick_magnitude=0.2):
    """
    Apply multiple random kicks, relax each, return best.
    """
    pos = positions.copy()
    try:
        energy, _ = dzugutov_energy_gradient(pos)
    except Exception:
        return np.inf, positions
    
    if not np.isfinite(energy):
        return np.inf, positions
    
    best_energy = energy
    best_pos = pos.copy()
    
    for _ in range(n_kicks):
        if not _check_time():
            break
        
        kick = rng.normal(0, kick_magnitude, size=pos.shape)
        new_pos = pos + kick
        
        dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue
        
        new_energy, new_relaxed = _l_bfgs_refine(new_pos, potential, parameter, max_iter=200)
        
        if np.isfinite(new_energy) and new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using multi-polyhedral seeds,
    basin hopping for global search, and L-BFGS for local refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Define seed strategies: (poly_type, scale)
    seed_configs = [
        ('ico', 0.95), ('ico', 1.0), ('ico', 1.05),
        ('dode', 0.95), ('dode', 1.0), ('dode', 1.05),
        ('cubo', 0.95), ('cubo', 1.0), ('cubo', 1.05),
        ('random', 1.0), ('random', 1.0), ('random', 1.0),
    ]
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    # Phase 1: Generate diverse seeds and relax
    candidates = []
    for i, (poly_type, scale) in enumerate(seed_configs):
        if not _check_time():
            break
        seed = seeds[i % len(seeds)]
        rng = np.random.default_rng(seed)
        
        try:
            pos = _generate_seed(size, rng, poly_type, scale=scale)
            if len(pos) == size:
                # Quick L-BFGS relaxation
                energy, positions = _l_bfgs_refine(pos, potential, parameter, max_iter=200)
                if np.isfinite(energy):
                    candidates.append((energy, positions.copy()))
        except Exception:
            pass
    
    # Add a few more random seeds with different seeds
    for extra_seed in [999, 1111, 2222, 3333]:
        if not _check_time():
            break
        rng = np.random.default_rng(extra_seed)
        try:
            pos = _generate_seed(size, rng, 'random', scale=1.0)
            if len(pos) == size:
                energy, positions = _l_bfgs_refine(pos, potential, parameter, max_iter=200)
                if np.isfinite(energy):
                    candidates.append((energy, positions.copy()))
        except Exception:
            pass
    
    if not candidates:
        # Fallback
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
        return best_positions
    
    # Sort by energy and take top candidates
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:min(len(candidates), 10)]
    
    # Phase 2: Basin hopping on top candidates
    for idx, (init_energy, init_pos) in enumerate(top_candidates):
        if not _check_time():
            break
        
        # Allocate more cycles to better candidates
        if idx == 0:
            n_cycles = 60
        elif idx < 3:
            n_cycles = 40
        else:
            n_cycles = 25
        
        rng_bh = np.random.default_rng(1000 + idx)
        
        try:
            bh_energy, bh_pos = _basin_hopping(
                init_pos, potential, parameter,
                rng=rng_bh, n_cycles=n_cycles, kick_scale=0.12
            )
        except Exception:
            continue
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos.copy()
    
    # Phase 3: Kick-and-relax on the best result for further improvement
    if best_positions is not None and _check_time():
        rng_kick = np.random.default_rng(5555)
        try:
            kick_energy, kick_pos = _kick_and_relax(
                best_positions, potential, parameter,
                rng=rng_kick, n_kicks=15, kick_magnitude=0.15
            )
            if kick_energy < best_energy:
                best_energy = kick_energy
                best_positions = kick_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS polishing
    if best_positions is not None and _check_time():
        try:
            final_energy, final_pos = _l_bfgs_refine(
                best_positions, potential, parameter, max_iter=300
            )
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    # Final fallback
    if best_positions is None:
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
