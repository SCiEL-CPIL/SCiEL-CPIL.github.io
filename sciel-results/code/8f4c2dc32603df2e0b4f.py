"""
Sutton-Chen cluster global minima: structure-prediction + basin hopping.
Strategy: seed starts from known icosahedral/FCC magic-number cores,
grow by adding atoms at high-symmetry surface sites, then basin-hopping
with adaptive step sizes and parallel multi-start.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 280.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _icosahedron_edges(verts):
    """Return list of edge pairs for the icosahedron."""
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # edge length threshold
                edges.append((i, j))
    return edges

def _icosahedron_faces(verts, edges):
    """Return list of face triples."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, 12):
                if (j, k) not in edge_set or (i, k) not in edge_set:
                    continue
                faces.append((i, j, k))
    return faces

def _build_icosahedral_cluster(n, seed=0):
    """
    Build an icosahedral cluster with shells:
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices) at radius R1
    Shell 2: 30 atoms (edge midpoints) at radius R2
    Shell 3: 20 atoms (face centers) at radius R3
    Total: 63 atoms. For n > 63, add outer shell atoms.
    """
    rng = np.random.default_rng(seed)
    verts = _get_icosahedron_vertices()
    edges = _icosahedron_edges(verts)
    faces = _icosahedron_faces(verts, edges)

    points = [np.zeros(3)]  # center

    # Shell 1: vertices at radius 1.0
    R1 = 1.0
    for v in verts:
        points.append(v * R1)

    # Shell 2: edge midpoints
    edge_len = np.linalg.norm(verts[0] - verts[1])
    R2 = np.sqrt(R1**2 - (edge_len / 2.0)**2)
    for (i, j) in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_dir = mid / np.linalg.norm(mid)
        points.append(mid_dir * R2)

    # Shell 3: face centers
    R3 = 0.0
    for (i, j, k) in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        R3 = np.linalg.norm(center)
        center_dir = center / R3
        points.append(center_dir * R3)

    points = np.array(points)

    if len(points) >= n:
        return points[:n]

    # For n > 63: add atoms on a shell beyond R3
    remaining = n - len(points)
    # Use icosahedral symmetry directions for the outer shell
    outer_dirs = []
    # Add vertex directions, edge-mid directions, face-center directions
    for v in verts:
        outer_dirs.append(v)
    for (i, j) in edges:
        mid = (verts[i] + verts[j]) / 2.0
        outer_dirs.append(mid / np.linalg.norm(mid))
    for (i, j, k) in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        outer_dirs.append(center / np.linalg.norm(center))

    R_outer = R3 * 1.3
    for idx in range(remaining):
        d = outer_dirs[idx % len(outer_dirs)]
        # Add slight jitter to break exact symmetry
        jitter = rng.normal(0, 0.02, 3)
        points = np.vstack([points, [d * R_outer + jitter]])

    return points[:n]

def _build_fcc_cluster(n, seed=0):
    """Build FCC cluster: take closest n atoms to origin in FCC lattice."""
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.3
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    pts = list(sorted_pts)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    return np.array(pts)

def _build_hcp_cluster(n, seed=0):
    """Build HCP cluster."""
    rng = np.random.default_rng(seed)
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    R = (3.0 * n / (2.0 / (np.sqrt(3) * a**2 * c)) / (4.0 * np.pi)) ** (1.0/3.0) * 1.3
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                x = ix * a + (iy % 2) * a * 0.5
                y = iy * a * np.sqrt(3) / 2
                z = iz * c
                points.append(np.array([x, y, z]))
                x_b = ix * a + (iy % 2) * a * 0.5 + a * 0.5
                y_b = iy * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                z_b = iz * c + c / 2
                points.append(np.array([x_b, y_b, z_b]))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    pts = list(sorted_pts)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    return np.array(pts)

def _build_bcc_cluster(n, seed=0):
    """Build BCC cluster."""
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (2.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.3
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    pts = list(sorted_pts)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    return np.array(pts)

def _build_random_compact(n, seed=0):
    """Random compact cluster via rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 3000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.45:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _perturb(positions, magnitude, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, magnitude, positions.shape)
    return positions + noise

def _basin_hopping_worker(start, kind, max_steps, seed):
    """
    Basin hopping with adaptive step size.
    Returns (best_energy, best_positions).
    """
    try:
        if _time_left() < 8.0:
            return None, None

        current_pos = start.copy()
        energy, pos = relax(current_pos, 'sutton_chen', kind)
        if not (np.all(np.isfinite(pos)) and _validate(pos)):
            return None, None

        best_energy = energy
        best_pos = pos.copy()
        step_size = 0.3

        for step in range(max_steps):
            if _time_left() < 12.0:
                break

            # Adaptive step size: decrease if improving, increase if stuck
            current_pos = _perturb(pos, step_size, seed=seed + step * 7 + 1)

            if not _validate(current_pos):
                current_pos = _perturb(pos, step_size * 0.3, seed=seed + step * 7 + 2)
                if not _validate(current_pos):
                    continue

            new_energy, new_pos = relax(current_pos, 'sutton_chen', kind)
            if not (np.all(np.isfinite(new_pos)) and _validate(new_pos)):
                continue

            dE = new_energy - energy
            if dE < 0:
                energy, pos = new_energy, new_pos
                step_size = max(step_size * 0.85, 0.05)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
            else:
                # Accept with Metropolis at T=0.5
                T = 0.5
                if np.random.default_rng(seed + step * 13).random() < np.exp(-dE / T):
                    energy, pos = new_energy, new_pos
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = pos.copy()
                step_size = min(step_size * 1.15, 0.8)

        return best_energy, best_pos
    except Exception:
        return None, None

def _local_search_from(positions, kind, seed):
    """
    More aggressive local search: multiple small perturbations + relax.
    """
    try:
        best_energy, best_pos = np.inf, None
        current_pos = positions.copy()

        for i in range(4):
            if _time_left() < 10.0:
                break
            p = _perturb(current_pos, 0.1, seed=seed + i * 3)
            if not _validate(p):
                continue
            e, pos = relax(p, 'sutton_chen', kind)
            if np.all(np.isfinite(pos)) and _validate(pos):
                if e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
                    current_pos = pos.copy()

        return best_energy, best_pos
    except Exception:
        return None, None

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = instance["n"]
    kind = instance.get("kind", "12-6")

    starts = []

    # 1. Icosahedral structure prediction (primary)
    for i in range(8):
        try:
            s = _build_icosahedral_cluster(n, seed=i)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    # 2. FCC
    for i in range(4):
        try:
            s = _build_fcc_cluster(n, seed=100 + i)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    # 3. HCP
    for i in range(3):
        try:
            s = _build_hcp_cluster(n, seed=200 + i)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    # 4. BCC
    for i in range(2):
        try:
            s = _build_bcc_cluster(n, seed=300 + i)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    # 5. Random compact
    for i in range(4):
        try:
            s = _build_random_compact(n, seed=400 + i)
            if len(s) == n and _validate(s):
                starts.append(s)
        except:
            pass

    if not starts:
        starts = [_build_random_compact(n, seed=0)]

    # Limit total starts to manage time
    max_starts = 22
    starts = starts[:max_starts]

    best_energy = np.inf
    best_positions = starts[0]

    # Parallel basin hopping
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, start in enumerate(starts):
            if _time_left() < 20.0:
                break
            # More steps for icosahedral starts (first 8), fewer for others
            max_steps = 8 if i < 8 else 5
            f = executor.submit(_basin_hopping_worker, start, kind, max_steps, i * 100)
            futures[f] = i

        for future in as_completed(futures):
            if _time_left() < 15.0:
                break
            try:
                res = future.result(timeout=60)
            except:
                continue
            if res[0] is not None:
                energy, positions = res
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()

    # Final polish: local search from best
    if _validate(best_positions) and _time_left() > 15.0:
        try:
            le, lp = _local_search_from(best_positions, kind, seed=9999)
            if le is not None and le < best_energy:
                best_energy = le
                best_positions = lp
        except:
            pass

    # One more relaxation
    if _validate(best_positions) and _time_left() > 8.0:
        try:
            fe, fp = relax(best_positions, 'sutton_chen', kind)
            if np.all(np.isfinite(fp)) and _validate(fp):
                if fe < best_energy:
                    best_energy = fe
                    best_positions = fp
        except:
            pass

    # Final validation
    if not _validate(best_positions):
        best_positions = _build_random_compact(n, seed=777)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36

    return best_positions
# EVOLVE-BLOCK-END
