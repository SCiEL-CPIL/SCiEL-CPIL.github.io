"""Structure prediction strategy: embed known small-cluster minima as cores, grow by adding atoms."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

def _fcc_positions(n, spacing=1.0):
    """Generate n positions in an FCC-like arrangement."""
    # FCC lattice vectors
    a = spacing
    # Generate points in a cube and keep those inside a sphere
    # Use a dense grid
    max_coord = int(np.ceil((n ** (1/3)) * a * 0.8)) + 2
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i * a + dx * a, j * a + dy * a, k * a + dz * a])
                    if np.linalg.norm(p) <= max_coord * a * 0.9:
                        points.append(p)
    points = np.array(points)
    # Sort by distance from origin and take n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _icosahedron_positions(n, spacing=1.0):
    """Generate positions using icosahedral symmetry for small clusters."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [-1, phi, 0]
    ])
    # Wait, that's not right. Let me use standard icosahedron vertices.
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0]) * spacing * 0.8
    
    if n <= 12:
        # Just use subset of icosahedron vertices
        if n == 1:
            return np.array([[0, 0, 0]])
        # Add center
        pts = [np.array([0, 0, 0])]
        for i in range(min(n - 1, 12)):
            pts.append(verts[i])
        return np.array(pts[:n])
    
    # For larger n, use nested shells
    pts = [np.array([0, 0, 0])]
    # Shell 1: 12 vertices at radius r1
    r1 = spacing * 0.8
    shell1 = verts * (r1 / np.linalg.norm(verts[0]))
    for v in shell1:
        pts.append(v)
        if len(pts) >= n:
            break
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Shell 2: 30 vertices at radius r2 (edges of icosahedron midpoints)
    r2 = spacing * 1.3
    # Generate edge midpoints of icosahedron
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < np.linalg.norm(verts[0]) * 1.2:  # Adjacent vertices
                mid = (verts[i] + verts[j]) / 2
                edges.append(mid)
    edges = np.array(edges)
    if len(edges) > 0:
        edges = edges / np.linalg.norm(edges[0]) * r2
        for v in edges:
            pts.append(v)
            if len(pts) >= n:
                break
    return np.array(pts[:n])

def _generate_initial_positions(n, kind, rng):
    """Generate multiple diverse initial configurations for relaxation."""
    configs = []
    
    # Configuration 1: FCC-like
    try:
        fcc = _fcc_positions(n, spacing=0.75)
        if len(fcc) == n and np.all(np.isfinite(fcc)):
            configs.append(fcc)
    except:
        pass
    
    # Configuration 2: Icosahedral
    try:
        ico = _icosahedron_positions(n, spacing=0.75)
        if len(ico) == n and np.all(np.isfinite(ico)):
            configs.append(ico)
    except:
        pass
    
    # Configuration 3: Random compact
    radius = 0.75 * n ** (1.0 / 3.0) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.4 for q in points):
                points.append(p)
        attempts += 1
    if len(points) == n:
        configs.append(np.array(points))
    
    # Configuration 4: Random with different seed
    radius2 = 0.72 * n ** (1.0 / 3.0) * 1.1
    points2 = []
    attempts2 = 0
    while len(points2) < n and attempts2 < n * 100:
        p = rng.uniform(-radius2, radius2, 3)
        if np.linalg.norm(p) <= radius2:
            if all(np.linalg.norm(p - q) >= 0.38 for q in points2):
                points2.append(p)
        attempts2 += 1
    if len(points2) == n:
        configs.append(np.array(points2))
    
    # Configuration 5: Spherical shell + core
    if n >= 13:
        core_n = min(13, n)
        shell_n = n - core_n
        core = _icosahedron_positions(core_n, spacing=0.7)
        shell_r = 0.7 * (n ** (1/3.0)) * 0.9
        shell_pts = []
        # Distribute shell points roughly uniformly on sphere
        for i in range(shell_n):
            # Use Fibonacci sphere for uniform distribution
            z = 1 - 2 * (i + 0.5) / shell_n
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / shell_n
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_r
            shell_pts.append(p)
        config5 = np.vstack([core, np.array(shell_pts)])
        if len(config5) == n and np.all(np.isfinite(config5)):
            configs.append(config5)
    
    return configs

def construct(instance):
    """Best of multiple structure-prediction initializations, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    rng = np.random.default_rng(size * 7 + 123)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    configs = _generate_initial_positions(size, kind, rng)
    
    # Add a few more random configurations
    for trial in range(3):
        radius = 0.73 * size ** (1.0 / 3.0) * (1.0 + 0.2 * trial)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 150:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            configs.append(np.array(points))
    
    # Relax each configuration
    for config in configs:
        if config is None or len(config) != size:
            continue
        if not np.all(np.isfinite(config)):
            continue
        # Check minimum distance
        min_dist = np.inf
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(config[i] - config[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            continue
        
        try:
            energy, positions = relax(config, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                # Verify validity
                if np.all(np.isfinite(positions)):
                    min_d = np.inf
                    for i in range(size):
                        for j in range(i + 1, size):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_d:
                                min_d = d
                    if min_d >= 0.35:
                        best_energy = energy
                        best_positions = positions
        except:
            pass
    
    # If no valid result, fall back to a simple random relaxation
    if best_positions is None:
        radius = 0.75 * size ** (1.0 / 3.0)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 200:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.4 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            try:
                energy, positions = relax(np.array(points), potential, kind)
                if np.all(np.isfinite(positions)):
                    best_positions = positions
            except:
                pass
    
    # Final fallback
    if best_positions is None:
        # Create a simple linear arrangement
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.5, 0, 0]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
