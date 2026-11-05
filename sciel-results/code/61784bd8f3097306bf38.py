"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _dzugutov_energy_flat(x):
    """Compute Dzugutov potential energy from flat array."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    return _dzugutov_energy(positions)


def _dzugutov_gradient_flat(x):
    """Compute gradient of Dzugutov potential energy from flat array."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            
            if r < a:
                # V = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A*(-m*r^(-m-1))*exp(c/(r-a)) + A*(r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                # V = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                dV_dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            # Force on i is -dV/dr * rij/r
            force = -dV_dr * rij / r
            grad[i * 3:(i + 1) * 3] += force
            grad[j * 3:(j + 1) * 3] -= force
    
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


def _get_icosahedron_faces():
    """Get the 20 triangular faces of a unit icosahedron."""
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
    
    # Edge length of unit icosahedron
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - edge_len) < 1e-6:
                edges.append((i, j))
    
    # Find all faces (triangles where all three edges exist)
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
    """Build a Mackay icosahedral cluster with n atoms."""
    # Mackay icosahedron: 1 center, 12 first shell, 42 second shell
    # Total: 1+12=13, 1+12+42=55
    
    # Optimal bond lengths for Dzugutov
    r1 = 1.0  # first shell radius
    r2 = 1.7  # second shell radius
    
    points = [np.zeros(3)]  # center
    
    # First shell: icosahedron vertices
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        # Select n points from center + icosahedron
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    # Second shell: 42 positions
    verts_unit, faces, edges = _get_icosahedron_faces()
    
    # Face centers (20 faces)
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    # Edge midpoints (30 edges)
    edge_mids = []
    for i, j in edges:
        em = (verts_unit[i] + verts_unit[j]) / 2
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
    
    # Vertex positions at larger radius (12 vertices)
    vertex_positions = list(verts_unit)
    
    # Second shell has 20 face centers + 30 edge midpoints = 50 candidates
    # But Mackay has 42: 20 face centers + 22 edge midpoints (some edges shared)
    # Actually, for Mackay: 12 vertices + 30 edge centers + 20 face centers = 62 positions
    # But the 42 second shell atoms are at specific positions
    
    # For simplicity, use face centers (20) + edge midpoints (30) = 50 candidates
    # Select 42 from these for full Mackay
    
    second_shell_candidates = []
    for fc in face_centers:
        second_shell_candidates.append(fc * r2)
    for em in edge_mids:
        second_shell_candidates.append(em * r2)
    
    # Add some vertex positions at intermediate radius
    for v in verts_unit:
        second_shell_candidates.append(v * (r1 + r2) / 2)
    
    second_shell_candidates = np.array(second_shell_candidates)
    
    # Select needed number of second shell atoms
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    # Shuffle and select
    idx = rng.permutation(len(second_shell_candidates))
    for i in range(min(needed, len(second_shell_candidates))):
        points.append(second_shell_candidates[idx[i]])
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
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


def _octahedron_vertices(radius=1.0):
    """Generate 6 vertices of a regular octahedron."""
    verts = np.array([
        [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0],
        [0, 0, radius], [0, 0, -radius]
    ], dtype=float)
    return verts


def _tetrahedron_vertices(radius=1.0):
    """Generate 4 vertices of a regular tetrahedron."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _build_icosahedral_cluster(n, rng):
    """Build a cluster based on icosahedral symmetry for n atoms."""
    # Use Mackay icosahedral structure
    return _build_mackay_icosahedron(n, rng)


def _build_random_compact(n, rng):
    """Build a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add them with small offsets
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.3 * rng.standard_normal(3)
        points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing."""
    # Start with a tetrahedron and add layers
    r = 1.0
    points = []
    
    # Central tetrahedron
    tet = _tetrahedron_vertices(r)
    points.extend(list(tet))
    
    # Add octahedron
    octa = _octahedron_vertices(r * 1.2)
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
    
    # Add more atoms in shells
    shell_radius = r * 1.5
    while len(points) < n:
        p = rng.uniform(-shell_radius, shell_radius, 3)
        if np.linalg.norm(p) <= shell_radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.2 * rng.standard_normal(3))
        else:
            points.append(p + 0.2 * rng.standard_normal(3))
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral initial guesses."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    # Strategy 1: Icosahedral Mackay structures with perturbations
    for i in range(5):
        if not check_time():
            break
        try:
            pos = _build_mackay_icosahedron(size, rng)
            # Vary perturbation scale
            scale = 0.05 + 0.05 * i
            pos = pos + scale * rng.standard_normal((size, 3))
            
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Strategy 2: Random compact structures
    for i in range(4):
        if not check_time():
            break
        try:
            pos = _build_random_compact(size, rng)
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Strategy 3: Tetrahedral structures
    for i in range(3):
        if not check_time():
            break
        try:
            pos = _build_tetrahedral_cluster(size, rng)
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Strategy 4: Local optimization around best solution with various perturbations
    if best_positions is not None:
        for i in range(6):
            if not check_time():
                break
            try:
                # Vary perturbation scale
                scale = 0.15 + 0.1 * i
                pos = best_positions + scale * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Strategy 5: Additional random restarts if time permits
    n_extra = 5
    for i in range(n_extra):
        if not check_time():
            break
        try:
            # Mix of strategies
            if i % 3 == 0:
                pos = _build_mackay_icosahedron(size, rng)
                scale = 0.1 + 0.05 * i
            elif i % 3 == 1:
                pos = _build_random_compact(size, rng)
                scale = 0.08
            else:
                pos = _build_tetrahedral_cluster(size, rng)
                scale = 0.08
            
            pos = pos + scale * rng.standard_normal((size, 3))
            
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Final: if we have time, do a more aggressive local search around best
    if best_positions is not None and check_time():
        for i in range(4):
            if not check_time():
                break
            try:
                # Larger perturbations to escape local minima
                scale = 0.3 + 0.1 * i
                pos = best_positions + scale * rng.standard_normal((size, 3))
                
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a simple structure
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
