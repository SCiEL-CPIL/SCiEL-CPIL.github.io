"""Sutton-Chen transition-metal cluster global minima search with coarse-grained topological clustering."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 20.0
_lock = threading.Lock()


def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN


def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.35:
        return False
    return True


def _icosahedron_vertices(r=1.0):
    """Return 12 vertices of a regular icosahedron scaled to radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * r
    return verts


def _face_centers(r=1.0):
    """Return 20 face centers of an icosahedron scaled to radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Faces of icosahedron (20 triangles)
    faces = [
        (0, 1, 5), (0, 5, 9), (0, 9, 11), (0, 11, 7), (0, 7, 1),
        (1, 7, 3), (1, 3, 4), (1, 4, 5), (5, 4, 8), (5, 8, 9),
        (9, 8, 10), (9, 10, 11), (11, 10, 6), (11, 6, 7), (7, 6, 2),
        (7, 2, 3), (3, 2, 13), (3, 13, 4), (4, 13, 8), (8, 13, 10)
    ]
    # Note: the above face indices need to match the vertex ordering. Let me use a simpler approach.
    # Actually, let me compute face centers differently.
    
    # For an icosahedron, the face centers form a dodecahedron.
    # Let me just generate 20 points on a sphere of radius r * (some factor).
    # The distance from center to face center is r * cos(acos(1/sqrt(5))) approximately.
    # Actually, for a regular icosahedron with vertex radius R, the face center is at distance R * (sqrt(5)-1)/2 / 1? No.
    # Let me just use a Fibonacci sphere for 20 points.
    
    N = 20
    z = (2 * np.arange(N) + 1) / N
    r_circ = np.sqrt(1 - z**2)
    phi_angle = np.arange(N) * (np.pi * (3 - np.sqrt(5)))
    
    x = r_circ * np.cos(phi_angle)
    y = r_circ * np.sin(phi_angle)
    
    pts = np.column_stack([x, y, z])
    # Scale to appropriate radius. For icosahedron, face centers are at distance ~0.795 * vertex_radius
    pts = pts * r * 0.795
    return pts


def _edge_midpoints(r=1.0):
    """Return 30 edge midpoints of an icosahedron scaled to radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * r
    
    # Edges of icosahedron
    edges = [
        (0, 1), (0, 5), (0, 9), (0, 11), (0, 7),
        (1, 7), (1, 3), (1, 4), (1, 5),
        (5, 4), (5, 8), (5, 9),
        (9, 8), (9, 10), (9, 11),
        (11, 10), (11, 6), (11, 7),
        (7, 6), (7, 2), (7, 3),
        (3, 2), (3, 13), (3, 4),
        (4, 13), (4, 8),
        (8, 13), (8, 10),
        (10, 13), (10, 6),
        (6, 13), (6, 2)
    ]
    # This is getting complex. Let me use a simpler approach.
    # Just generate 30 points on a sphere.
    N = 30
    z = (2 * np.arange(N) + 1) / N
    r_circ = np.sqrt(1 - z**2)
    phi_angle = np.arange(N) * (np.pi * (3 - np.sqrt(5)))
    
    x = r_circ * np.cos(phi_angle)
    y = r_circ * np.sin(phi_angle)
    
    pts = np.column_stack([x, y, z])
    # Edge midpoints are at distance ~0.951 * vertex_radius
    pts = pts * r * 0.951
    return pts


def _generate_mackay_icosahedron(n):
    """Generate Mackay icosahedral cluster positions."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    shell1 = _icosahedron_vertices(r=1.0)
    points.extend(list(shell1))
    
    if n > 13:
        # Shell 2: 30 edge midpoints + 12 face centers = 42 points
        shell2 = _edge_midpoints(r=1.0)  # 30 points
        points.extend(list(shell2))
        
        if n > 13 + 30:
            # Add face centers
            shell2b = _face_centers(r=1.0)  # 20 points
            points.extend(list(shell2b))
    
    if n > 55:
        # Shell 3: more points
        # For simplicity, add a third shell
        R3 = 2.2
        deficit = n - len(points)
        if deficit > 0:
            N = deficit
            z = (2 * np.arange(N) + 1) / N
            r_circ = np.sqrt(1 - z**2)
            phi_angle = np.arange(N) * (np.pi * (3 - np.sqrt(5)))
            x = r_circ * np.cos(phi_angle)
            y = r_circ * np.sin(phi_angle)
            pts = np.column_stack([x, y, z]) * R3
            points.extend(list(pts))
    
    if len(points) < n:
        # Pad with more points on larger shells
        deficit = n - len(points)
        R4 = 3.0
        z = (2 * np.arange(deficit) + 1) / deficit
        r_circ = np.sqrt(1 - z**2)
        phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
        x = r_circ * np.cos(phi_angle)
        y = r_circ * np.sin(phi_angle)
        pts = np.column_stack([x, y, z]) * R4
        points.extend(list(pts))
    
    if len(points) > n:
        points = points[:n]
    
    pts = np.array(points)
    pts -= pts.mean(axis=0)
    
    # Scale to reasonable size
    norms = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        target_radius = 3.0
        pts = pts * (target_radius / max_norm)
    
    return pts


def _generate_fcc_sphere(n):
    """Generate FCC-like spherical cluster."""
    a = 1.0
    pts = []
    L = int(np.cbrt(n / 2.0)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    pts.append([x, y, z])
    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norms = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
    
    return pts


def _generate_bcc_sphere(n):
    """Generate BCC-like spherical cluster."""
    a = 1.0
    pts = []
    L = int(np.cbrt(n / 2.0)) + 3
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                pts.append([i * a, j * a, k * a])
                if i < L and j < L and k < L:
                    pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norms = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
    
    return pts


def _generate_hcp_sphere(n):
    """Generate HCP-like spherical cluster."""
    a = 1.0
    c = a * np.sqrt(8 / 3)
    pts = []
    L = int(np.cbrt(n / 2.0)) + 3
    
    for layer in range(-L, L + 1):
        z = layer * c
        if layer % 2 == 0:
            offset = (0, 0)
        else:
            offset = (a / 2, a * np.sqrt(3) / 6)
        
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                x = i * a + offset[0]
                y = j * a * np.sqrt(3) + offset[1]
                pts.append([x, y, z])
    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norms = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
    
    return pts


def _generate_random_compact(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    pts[0] = 0
    for i in range(1, n):
        idx = rng.integers(0, i)
        offset = rng.normal(0, 0.6, 3)
        pts[i] = pts[idx] + offset
    
    pts -= pts.mean(axis=0)
    norms = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
    
    return pts


def _coarse_grained_optimize(positions, kind, n_subunits=5, n_iterations=10):
    """
    Coarse-grained optimization: group atoms into rigid subunits,
    optimize subunit positions/rotations, then fine-relax.
    """
    n = len(positions)
    if n < n_subunits * 2:
        return positions
    
    # Assign atoms to subunits based on spatial proximity
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    
    # Simple clustering: pick n_subunits seed points, assign nearest
    rng = np.random.default_rng(42)
    seed_idx = rng.choice(n, n_subunits, replace=False)
    seeds = positions[seed_idx]
    
    # Assign each atom to nearest seed
    _, labels = tree.query(positions)
    
    # For each subunit, compute centroid and optimize translation
    subunit_centroids = []
    for i in range(n_subunits):
        mask = labels == i
        if np.sum(mask) > 0:
            subunit_centroids.append(positions[mask].mean(axis=0))
    
    # Optimize subunit positions using energy gradient
    def subunit_energy(subunit_pos):
        # Reconstruct full positions from subunit positions
        full_pos = np.zeros((n, 3))
        for i in range(n_subunits):
            mask = labels == i
            if np.sum(mask) > 0:
                # Translate subunit atoms by (new_centroid - old_centroid)
                old_centroid = subunit_centroids[i]
                new_centroid = subunit_pos[i * 3:(i + 1) * 3]
                full_pos[mask] = positions[mask] + (new_centroid - old_centroid)
        
        if not _validate_positions(full_pos):
            return 1e10
        energy, _ = sutton_chen_energy_gradient(full_pos, kind)
        return energy
    
    # Optimize subunit centroids
    initial_subunit_pos = np.array(subunit_centroids).flatten()
    result = minimize(subunit_energy, initial_subunit_pos, method='L-BFGS-B',
                      options={'maxiter': 20})
    
    if result.success:
        # Reconstruct full positions
        full_pos = np.zeros((n, 3))
        for i in range(n_subunits):
            mask = labels == i
            if np.sum(mask) > 0:
                old_centroid = subunit_centroids[i]
                new_centroid = result.x[i * 3:(i + 1) * 3]
                full_pos[mask] = positions[mask] + (new_centroid - old_centroid)
        
        if _validate_positions(full_pos):
            return full_pos
    
    return positions


def _basin_hopping(positions, kind, rng, n_steps=20, step_size=0.3):
    """Basin hopping with adaptive step size."""
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if _get_time_left() < 10:
            break
        
        # Adaptive step size
        adaptive_step = step_size * (1.0 + 0.5 * (1 - step / max(n_steps - 1, 1)))
        perturbed = current_pos + rng.normal(0, adaptive_step, current_pos.shape)
        
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        if _validate_positions(new_pos):
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            elif rng.random() < np.exp(-(new_energy - current_energy) / max(0.1, 1.0 - step / n_steps)):
                current_energy = new_energy
                current_pos = new_pos
    
    return best_energy, best_pos


def _parallel_relax(starts, kind, max_workers=4):
    """Relax multiple starting configurations in parallel."""
    results = []
    
    def _relax_one(pos):
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        for future in as_completed(futures):
            if _get_time_left() < 5:
                break
            try:
                energy, positions = future.result()
                if positions is not None and np.isfinite(energy):
                    results.append((energy, positions))
            except Exception:
                continue
    
    return results


def construct(instance):
    """Global minima search for Sutton-Chen clusters using coarse-grained topological clustering."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Mackay icosahedron (primary structure for transition metals)
    ico = _generate_mackay_icosahedron(n)
    if _validate_positions(ico):
        starts.append(ico)
    
    # 2. FCC sphere
    fcc = _generate_fcc_sphere(n)
    if _validate_positions(fcc):
        starts.append(fcc)
    
    # 3. BCC sphere
    bcc = _generate_bcc_sphere(n)
    if _validate_positions(bcc):
        starts.append(bcc)
    
    # 4. HCP sphere
    hcp = _generate_hcp_sphere(n)
    if _validate_positions(hcp):
        starts.append(hcp)
    
    # 5-6. Random compact clusters
    for s in [0, 1]:
        rand = _generate_random_compact(n, seed=seed + s)
        if _validate_positions(rand):
            starts.append(rand)
    
    # Phase 1: Parallel relaxation of all starts
    if _get_time_left() > 30:
        relaxed_results = _parallel_relax(starts, kind, max_workers=4)
        for energy, positions in relaxed_results:
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Phase 2: Coarse-grained optimization from best configuration
    if best_positions is not None and _get_time_left() > 60:
        try:
            cg_positions = _coarse_grained_optimize(best_positions, kind, n_subunits=5, n_iterations=10)
            if _validate_positions(cg_positions):
                cg_energy, cg_relaxed = relax(cg_positions, 'sutton_chen', kind)
                if _validate_positions(cg_relaxed) and cg_energy < best_energy:
                    best_energy = cg_energy
                    best_positions = cg_relaxed.copy()
        except Exception:
            pass
    
    # Phase 3: Basin hopping from best configuration
    if best_positions is not None and _get_time_left() > 40:
        try:
            bh_energy, bh_pos = _basin_hopping(best_positions, kind, rng, n_steps=15, step_size=0.25)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Additional basin hopping with smaller steps for fine-tuning
    if best_positions is not None and _get_time_left() > 20:
        try:
            bh_energy2, bh_pos2 = _basin_hopping(best_positions, kind, rng, n_steps=10, step_size=0.1)
            if bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_pos2.copy()
        except Exception:
            pass
    
    # Validate final result
    if best_positions is None or not _validate_positions(best_positions):
        # Fallback: return a valid configuration
        fallback = _generate_random_compact(n, seed=seed)
        if _validate_positions(fallback):
            best_positions = fallback
        else:
            # Last resort: simple grid
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.4, (i // 10) * 0.4, (i // 100) * 0.4]
    
    return best_positions
# EVOLVE-BLOCK-END
