"""Dzugutov cluster global minima search using topological optimization with barrier-augmented L-BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize, brentq
from scipy.spatial import cKDTree


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _find_r_min():
    """Find r_min where dV/dr = 0 in region 1 (r < a)."""
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
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    
    energy = 0.0
    if len(r) == 0:
        return energy
    
    r_safe = np.maximum(r, 1e-10)
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r_safe[mask1]
        energy += np.sum(A * (r1**(-m) - B) * np.exp(c / (r1 - a)))
    
    # Region 2: a <= r < b
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
    """Compute gradient of Dzugutov potential energy from flat array."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    r_safe = np.maximum(r, 1e-10)
    
    for idx in range(len(r)):
        i, j = iu[0][idx], iu[1][idx]
        ri = r_safe[idx]
        
        if ri < a:
            exp_term = np.exp(c / (ri - a))
            dVdr = A * (-m * ri**(-m-1)) * exp_term + \
                   A * (ri**(-m) - B) * exp_term * (-c / (ri - a)**2)
        elif ri < b:
            dVdr = B * np.exp(d / (ri - b)) * (-d / (ri - b)**2)
        else:
            continue
        
        factor = -dVdr / ri
        grad[3*i:3*i+3] += factor * diff[i, j]
        grad[3*j:3*j+3] -= factor * diff[i, j]
    
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
    """Build a Mackay icosahedral cluster with n atoms using optimal radii."""
    r1 = R_MIN  # first shell radius at potential minimum
    r2 = r1 * 1.72  # second shell radius
    
    points = [np.zeros(3)]  # center
    
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
    
    second_shell = []
    for fc in face_centers:
        second_shell.append(fc * r2)
    for em in edge_mids:
        second_shell.append(em * r2)
    
    second_shell = np.array(second_shell)
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    n_select = min(needed, len(second_shell))
    idx = rng.permutation(len(second_shell))
    for i in range(n_select):
        points.append(second_shell[idx[i]])
    
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.15 * rng.standard_normal(3))
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster with polytetrahedral bias."""
    spacing = R_MIN
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 15000:
        if rng.random() < 0.3 and len(points) >= 13:
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
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
            p = base + 0.35 * rng.standard_normal(3)
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
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
    
    shell_radius = r * 1.6
    while len(points) < n:
        if rng.random() < 0.4 and len(points) > 4:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + r * 0.9 * offset_dir
        else:
            p = rng.uniform(-shell_radius, shell_radius, 3)
            if np.linalg.norm(p) > shell_radius:
                p = p * shell_radius / np.linalg.norm(p)
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            points.append(p + 0.2 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _local_optimize(pos, potential, parameter, time_budget=5.0):
    """Run local optimization using relax and L-BFGS-B."""
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
    if time.time() - start < time_budget * 0.4:
        try:
            x0 = best_pos.flatten()
            result = minimize(
                _dzugutov_energy_flat,
                x0,
                jac=_dzugutov_gradient_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3).copy()
        except Exception:
            pass
    
    # Final relax from L-BFGS-B result
    if time.time() - start < time_budget * 0.8:
        try:
            energy, positions = relax(best_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov clusters using topological optimization."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 25.0
    
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
        time_budget = max(2.0, min(8.0, 300.0 / max(1, (time_limit - safety_margin) / 20)))
        try:
            energy, positions = _local_optimize(pos, potential, parameter, time_budget)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 1: Mackay icosahedral structures with varied perturbations
    n_ico_tries = min(10, max(4, int(40 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_ico_tries):
        if not check_time():
            break
        pos = _build_mackay_icosahedron(size, rng)
        scale = 0.02 + 0.03 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_optimize(pos, f"ico_{i}")
    
    # Strategy 2: Random compact structures with polytetrahedral bias
    n_rand_tries = min(8, max(3, int(30 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_rand_tries):
        if not check_time():
            break
        pos = _build_random_compact(size, rng)
        pos = pos + 0.03 * rng.standard_normal((size, 3))
        try_optimize(pos, f"rand_{i}")
    
    # Strategy 3: Tetrahedral structures
    n_tet_tries = min(6, max(2, int(20 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_tet_tries):
        if not check_time():
            break
        pos = _build_tetrahedral_cluster(size, rng)
        pos = pos + 0.03 * rng.standard_normal((size, 3))
        try_optimize(pos, f"tet_{i}")
    
    # Strategy 4: Local optimization around best solution with varied perturbations
    if best_positions is not None:
        n_local_tries = min(12, max(5, int(40 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_local_tries):
            if not check_time():
                break
            scale = 0.05 + 0.05 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_optimize(pos, f"local_{i}")
    
    # Strategy 5: Additional diverse restarts if time permits
    n_extra = min(10, max(4, int(35 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_extra):
        if not check_time():
            break
        if i % 3 == 0:
            pos = _build_mackay_icosahedron(size, rng)
            scale = 0.10 + 0.03 * (i // 3)
        elif i % 3 == 1:
            pos = _build_random_compact(size, rng)
            scale = 0.06 + 0.03 * (i // 3)
        else:
            pos = _build_tetrahedral_cluster(size, rng)
            scale = 0.06 + 0.03 * (i // 3)
        
        pos = pos + scale * rng.standard_normal((size, 3))
        try_optimize(pos, f"extra_{i}")
    
    # Strategy 6: Aggressive local search around best with larger perturbations
    if best_positions is not None and check_time():
        n_aggressive = min(8, max(3, int(25 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_aggressive):
            if not check_time():
                break
            scale = 0.20 + 0.08 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_optimize(pos, f"agg_{i}")
    
    # Strategy 7: Fine-tuning around best
    if best_positions is not None and check_time():
        n_fine = min(5, max(2, int(15 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_fine):
            if not check_time():
                break
            scale = 0.01 + 0.008 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_optimize(pos, f"fine_{i}")
    
    if best_positions is None:
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
