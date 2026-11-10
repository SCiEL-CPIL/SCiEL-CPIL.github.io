# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

def _remaining():
    return _DEADLINE - time.time()

def _get_spacing(kind):
    """Equilibrium bond length from pair potential minimum."""
    if kind == '12-6':
        n, m, c = 12, 6, 144.41
    elif kind == '9-6':
        n, m, c = 9, 6, 39.432
    elif kind == '10-8':
        n, m, c = 10, 8, 34.408
    else:
        return 1.0
    return (2.0 * n / (c * m)) ** (1.0 / (n - m / 2.0))

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _icosahedron_shell2():
    """Return 42 vertices of the second icosahedral shell (truncated icosahedron-like).
    These are the midpoints of edges of the icosahedron, plus additional points.
    Actually, the 42-atom second shell consists of:
    - 30 edge midpoints of the icosahedron
    - 12 face centers
    But that's 42. Let me construct them properly.
    """
    # Get icosahedron vertices
    verts = _icosahedron_vertices()
    
    # Find all edges (pairs of vertices at minimum distance)
    n = len(verts)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Minimum distance is the edge length
    min_dist = np.min(dists[dists > 0])
    tol = 1e-6
    
    # Edge midpoints (30 edges)
    edge_mids = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[i, j] - min_dist) < tol:
                mid = (verts[i] + verts[j]) / 2.0
                edge_mids.append(mid)
    
    # Face centers: for each face (triangle of 3 mutually connected vertices)
    face_centers = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(dists[i, j] - min_dist) < tol:
                for k in range(j+1, n):
                    if (abs(dists[i, k] - min_dist) < tol and 
                        abs(dists[j, k] - min_dist) < tol):
                        fc = (verts[i] + verts[j] + verts[k]) / 3.0
                        face_centers.append(fc)
    
    all_pts = np.array(edge_mids + face_centers)
    # Normalize to unit sphere
    norms = np.linalg.norm(all_pts, axis=1, keepdims=True)
    return all_pts / norms

def _icosahedral_cluster(n, spacing, center_at_origin=True):
    """Generate an icosahedral cluster with n atoms.
    n=1: just center
    n=13: center + 12 vertices
    n=55: center + 12 + 42
    For other n, truncate or add atoms.
    """
    points = []
    
    if n >= 1:
        points.append(np.zeros(3))
    
    if n >= 13:
        verts = _icosahedron_vertices() * spacing
        points.extend([v.copy() for v in verts])
    
    if n >= 55:
        shell2 = _icosahedron_shell2() * spacing
        points.extend([v.copy() for v in shell2])
    
    # If we need more than 55, add more atoms on a larger shell
    while len(points) < n:
        # Add atoms at the tips of the icosahedron, slightly outward
        verts = _icosahedron_vertices() * (spacing * 1.3)
        for v in verts:
            if len(points) >= n:
                break
            # Check if this position is not too close to existing points
            if all(np.linalg.norm(v - np.array(p)) > spacing * 0.5 for p in points):
                points.append(v.copy())
    
    # If we have too many, truncate
    points = points[:n]
    
    return np.array(points)

def _fcc_positions(n, a=1.0):
    """Generate n atoms in FCC lattice (truncated sphere)."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    sites = []
    max_idx = int(np.ceil(n ** (1.0/3.0)) + 3)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    
    positions = []
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 8))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    """Generate n atoms in HCP lattice (truncated)."""
    c = a * np.sqrt(8.0/3.0)
    sites = []
    max_idx = int(np.ceil(n ** (1.0/3.0)) + 3)
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2.0
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3.0)/2.0, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1.0/6.0) * a * np.sqrt(3.0), z + c/2.0])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    
    positions = []
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 8))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _bcc_positions(n, a=1.0):
    """Generate n atoms in BCC lattice (truncated)."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a
    sites = []
    max_idx = int(np.ceil(n ** (1.0/3.0)) + 3)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    
    positions = []
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 8))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    """Random rejection-sampling cluster."""
    if min_dist is None:
        min_dist = spacing * 0.85
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.1
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        # Fill remaining with random points (may cause close pairs, but relax will fix)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _check_valid(pos):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    # Compute pairwise distances
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    spacing = _get_spacing(kind)
    
    # Generate diverse candidates
    candidates = []
    
    # Icosahedral (especially good for n=55, 59)
    try:
        ico = _icosahedral_cluster(n, spacing)
        if len(ico) == n:
            candidates.append(("ico", ico))
    except Exception:
        pass
    
    # FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(("fcc", fcc))
    except Exception:
        pass
    
    # HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(("hcp", hcp))
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n:
            candidates.append(("bcc", bcc))
    except Exception:
        pass
    
    # Perturbed versions
    for base_name, base_pos in list(candidates[:]):
        for kick in [0.05, 0.1, 0.15]:
            rng = np.random.default_rng(hash((base_name, kick)) % 2**31)
            noise = rng.normal(0, kick * spacing, base_pos.shape)
            candidates.append((f"{base_name}_p{kick}", base_pos + noise))
    
    # Random clusters
    for seed in [42, 123, 456, 789, 101, 202]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append((f"rand_{seed}", rand))
        except Exception:
            pass
    
    # Relax all candidates, keep best
    best_energy = np.inf
    best_positions = None
    
    for name, pos in candidates:
        if _remaining() < 12.0:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if _check_valid(relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass
    
    # If we have time, try more perturbations of the best solution
    if best_positions is not None and _remaining() > 25.0:
        rng = np.random.default_rng(777)
        for kick in [0.05, 0.1, 0.15, 0.2, 0.3]:
            if _remaining() < 12.0:
                break
            try:
                noise = rng.normal(0, kick * spacing, best_positions.shape)
                perturbed = best_positions + noise
                energy, relaxed_pos = relax(perturbed, potential, kind)
                if _check_valid(relaxed_pos) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass
    
    # Fallback
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 99999, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _check_valid(relaxed_pos):
                best_positions = relaxed_pos
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.5, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
