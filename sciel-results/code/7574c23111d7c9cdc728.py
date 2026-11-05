"""Dzugutov cluster global minima search using Voronoi topology-constrained basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    from scipy.optimize import brentq
    def f(r):
        return -m / r - (r**(-m) - B) * c / (r - a)**2
    try:
        return brentq(f, 0.8, 1.86, xtol=1e-10)
    except:
        return 1.05

R_MIN = _find_r_min()


def _dzugutov_energy_vectorized(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    energy = 0.0
    if len(r) == 0:
        return energy
    r_safe = np.maximum(r, 1e-10)
    mask1 = r < a
    if np.any(mask1):
        r1 = r_safe[mask1]
        energy += np.sum(A * (r1**(-m) - B) * np.exp(c / (r1 - a)))
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r_safe[mask2]
        energy += np.sum(B * np.exp(d / (r2 - b)))
    return energy


def _dzugutov_energy_flat(x):
    """Compute Dzugutov potential energy from flat array."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    return _dzugutov_energy_vectorized(positions)


def _dzugutov_gradient_flat(x):
    """Compute gradient of Dzugutov potential energy from flat array (vectorized)."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    r_safe = np.maximum(r, 1e-10)
    
    dVdr = np.zeros_like(r)
    mask1 = r < a
    mask2 = (r >= a) & (r < b)
    
    if np.any(mask1):
        r1 = r_safe[mask1]
        dVdr[mask1] = A * (-m * r1**(-m-1)) * np.exp(c / (r1 - a)) + \
                      A * (r1**(-m) - B) * np.exp(c / (r1 - a)) * (-c / (r1 - a)**2)
    
    if np.any(mask2):
        r2 = r_safe[mask2]
        dVdr[mask2] = B * np.exp(d / (r2 - b)) * (-d / (r2 - b)**2)
    
    factor = -dVdr / r_safe
    for idx in range(len(r)):
        i, j = iu[0][idx], iu[1][idx]
        f = factor[idx]
        if abs(f) > 1e-15:
            grad[3*i:3*i+3] += f * diff[i, j]
            grad[3*j:3*j+3] -= f * diff[i, j]
    
    return grad


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
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


def _get_icosahedron_topology():
    """Get faces and edges of unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(dist - edge_len) < 1e-6:
                edges.append((i, j))
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (tuple(sorted((i, j))) in edge_set and
                    tuple(sorted((j, k))) in edge_set and
                    tuple(sorted((i, k))) in edge_set):
                    faces.append((i, j, k))
    return verts_unit, faces, edges


def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster with n atoms using r_min spacing."""
    r1 = R_MIN
    r2 = 2.0 * R_MIN
    
    points = [np.zeros(3)]
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= len(points):
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    verts_unit, faces, edges = _get_icosahedron_topology()
    
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    edge_mids = []
    for i, j in edges:
        em = (verts_unit[i] + verts_unit[j]) / 2
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
    
    second_shell_candidates = []
    for fc in face_centers:
        second_shell_candidates.append((np.linalg.norm(fc * r2), fc * r2))
    for em in edge_mids:
        second_shell_candidates.append((np.linalg.norm(em * r2), em * r2))
    
    second_shell_candidates.sort(key=lambda x: x[0])
    
    needed = n - len(points)
    for _, p in second_shell_candidates:
        if len(points) >= n:
            break
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
    
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        phi_angle = np.arccos(2 * rng.random() - 1)
        rad = rng.uniform(r1, r2 * 1.2)
        p = np.array([
            rad * np.sin(phi_angle) * cos(theta),
            rad * np.sin(phi_angle) * np.sin(theta),
            rad * np.cos(phi_angle)
        ])
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster with polytetrahedral bias using r_min spacing."""
    spacing = R_MIN * 1.05
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        if rng.random() < 0.4 and len(points) >= 13:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + spacing * offset_dir
        else:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
            p = base + 0.3 * rng.standard_normal(3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing (polytetrahedral model)."""
    r = R_MIN
    points = []
    
    tet = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(tet, axis=1)
    tet = tet / norms[:, np.newaxis] * r
    points.extend(list(tet))
    
    octa = np.array([
        [r * 1.3, 0, 0], [-r * 1.3, 0, 0],
        [0, r * 1.3, 0], [0, -r * 1.3, 0],
        [0, 0, r * 1.3], [0, 0, -r * 1.3]
    ], dtype=float)
    for p in octa:
        if len(points) >= n:
            break
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
    
    shell_radius = r * 1.6
    while len(points) < n:
        if rng.random() < 0.5 and len(points) > 4:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + r * 0.95 * offset_dir
        else:
            p = rng.uniform(-shell_radius, shell_radius, 3)
            if np.linalg.norm(p) > shell_radius:
                p = p * shell_radius / np.linalg.norm(p)
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            points.append(p + 0.15 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _voronoi_topology_loss(x, target_coordination=12, target_volume=None):
    """
    Compute a topology loss based on Voronoi cell properties.
    Penalizes deviations from ideal polytetrahedral coordination.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    
    # Add boundary points to make Voronoi finite
    max_coord = np.max(np.abs(positions)) + 5.0
    boundary_points = np.array([
        [max_coord, 0, 0], [-max_coord, 0, 0],
        [0, max_coord, 0], [0, -max_coord, 0],
        [0, 0, max_coord], [0, 0, -max_coord],
        [max_coord, max_coord, 0], [-max_coord, -max_coord, 0],
        [max_coord, -max_coord, 0], [-max_coord, max_coord, 0],
        [0, max_coord, max_coord], [0, -max_coord, -max_coord],
        [max_coord, 0, max_coord], [-max_coord, 0, -max_coord],
        [max_coord, max_coord, max_coord], [-max_coord, -max_coord, -max_coord],
        [max_coord, -max_coord, max_coord], [-max_coord, max_coord, -max_coord],
        [max_coord, max_coord, -max_coord], [-max_coord, -max_coord, max_coord],
        [max_coord, -max_coord, -max_coord], [-max_coord, max_coord, max_coord],
        [max_coord, max_coord, max_coord], [-max_coord, -max_coord, -max_coord],
    ])
    
    all_points = np.vstack([positions, boundary_points])
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        return 0.0
    
    loss = 0.0
    
    # For each atom, count neighbors within a cutoff (proxy for Voronoi faces)
    cutoff = 1.5 * R_MIN
    tree = cKDTree(positions)
    
    for i in range(n):
        neighbors = tree.query_ball_point(positions[i], cutoff)
        n_neighbors = len(neighbors) - 1  # exclude self
        
        # Penalize deviation from target coordination
        coord_penalty = (n_neighbors - target_coordination) ** 2
        loss += coord_penalty
    
    return loss / n


def _voronoi_constrained_energy(x, topology_weight=2.0):
    """Energy with Voronoi topology constraint."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    base_energy = _dzugutov_energy_vectorized(positions)
    topo_loss = _voronoi_topology_loss(x)
    return base_energy + topology_weight * topo_loss


def _local_optimize(pos, potential, parameter, time_budget=5.0):
    """Run local optimization using both relax and L-BFGS-B."""
    best_energy = float('inf')
    best_pos = pos.copy()
    start = time.time()
    
    # First try the built-in relax
    try:
        energy, positions = relax(pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = positions.copy()
    except Exception:
        pass
    
    # Then try L-BFGS-B if time permits
    if time.time() - start < time_budget * 0.5:
        try:
            x0 = best_pos.flatten()
            result = minimize(
                _dzugutov_energy_flat,
                x0,
                jac=_dzugutov_gradient_flat,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3).copy()
        except Exception:
            pass
    
    # Final relax from L-BFGS-B result
    if time.time() - start < time_budget * 0.9:
        try:
            energy, positions = relax(best_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    return best_energy, best_pos


def _voronoi_guided_optimize(pos, potential, parameter, time_budget=5.0, topology_weight=2.0):
    """
    Optimization guided by Voronoi topology constraints.
    Uses the topology loss to bias the search toward polytetrahedral structures.
    """
    best_energy = float('inf')
    best_pos = pos.copy()
    start = time.time()
    
    # First, try standard relax
    try:
        energy, positions = relax(pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = positions.copy()
    except Exception:
        pass
    
    # Then, try topology-constrained optimization
    if time.time() - start < time_budget * 0.5:
        try:
            x0 = best_pos.flatten()
            
            # Use a combined objective that includes topology penalty
            def combined_obj(x):
                return _voronoi_constrained_energy(x, topology_weight)
            
            # Numerical gradient for the combined objective
            def combined_grad(x):
                # Use central differences for the topology part
                eps = 1e-6
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    grad[i] = (combined_obj(x_plus) - combined_obj(x_minus)) / (2 * eps)
                return grad
            
            result = minimize(
                combined_obj,
                x0,
                jac=combined_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Evaluate true energy (without topology penalty)
            true_energy = _dzugutov_energy_vectorized(result.x.reshape(-1, 3))
            if true_energy < best_energy:
                best_energy = true_energy
                best_pos = result.x.reshape(-1, 3).copy()
        except Exception:
            pass
    
    # Final relax
    if time.time() - start < time_budget * 0.9:
        try:
            energy, positions = relax(best_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using Voronoi-topology-constrained basin hopping."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 30.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 7 + 13)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_optimize(pos, label=""):
        nonlocal best_energy, best_positions
        if not check_time():
            return
        time_budget = max(2.0, min(10.0, (time_limit - safety_margin) * 0.08))
        try:
            energy, positions = _local_optimize(pos, potential, parameter, time_budget)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    def try_voronoi_optimize(pos, label=""):
        """Optimize with Voronoi topology guidance."""
        nonlocal best_energy, best_positions
        if not check_time():
            return
        time_budget = max(2.0, min(10.0, (time_limit - safety_margin) * 0.08))
        try:
            energy, positions = _voronoi_guided_optimize(pos, potential, parameter, time_budget)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Phase 1: Build diverse initial structures
    # Strategy 1: Mackay icosahedral structures
    n_ico_tries = min(8, max(4, int(30 * (time_limit - safety_margin) / time_limit)))
    for i in range(n_ico_tries):
        if not check_time():
            break
        pos = _build_mackay_icosahedron(size, rng)
        scale = 0.01 + 0.02 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_optimize(pos, f"ico_{i}")
    
    # Strategy 2: Random compact structures
    n_rand_tries = min(6, max(3, int(20 * (time_limit - safety_margin) / time_limit)))
    for i in range(n_rand_tries):
        if not check_time():
            break
        pos = _build_random_compact(size, rng)
        pos = pos + 0.02 * rng.standard_normal((size, 3))
        try_optimize(pos, f"rand_{i}")
    
    # Strategy 3: Tetrahedral structures
    n_tet_tries = min(5, max(2, int(15 * (time_limit - safety_margin) / time_limit)))
    for i in range(n_tet_tries):
        if not check_time():
            break
        pos = _build_tetrahedral_cluster(size, rng)
        pos = pos + 0.02 * rng.standard_normal((size, 3))
        try_optimize(pos, f"tet_{i}")
    
    # Phase 2: Basin hopping around best solution
    if best_positions is not None:
        n_basin = min(15, max(8, int(40 * (time_limit - safety_margin) / time_limit)))
        for i in range(n_basin):
            if not check_time():
                break
            progress = i / max(1, n_basin)
            scale = 0.3 * (1.0 - progress) + 0.05
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_optimize(pos, f"basin_{i}")
    
    # Phase 3: Voronoi-guided refinement
    if best_positions is not None:
        n_voronoi = min(10, max(5, int(25 * (time_limit - safety_margin) / time_limit)))
        for i in range(n_voronoi):
            if not check_time():
                break
            scale = 0.05 + 0.03 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_voronoi_optimize(pos, f"voronoi_{i}")
    
    # Phase 4: Final fine-tuning
    if best_positions is not None:
        n_fine = min(8, max(4, int(20 * (time_limit - safety_margin) / time_limit)))
        for i in range(n_fine):
            if not check_time():
                break
            scale = 0.01 + 0.005 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_optimize(pos, f"fine_{i}")
    
    # Phase 5: Final relax
    if best_positions is not None and check_time():
        try:
            energy, positions = relax(best_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    if best_positions is None:
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
