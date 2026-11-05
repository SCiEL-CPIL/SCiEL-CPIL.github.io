"""Dzugutov cluster global minima search using diverse polytetrahedral structures."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

# Global time tracker
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    r2 = r2[i_idx, j_idx]
    r = np.sqrt(r2)
    valid = r > 1e-12
    r = r[valid]
    if len(r) == 0:
        return 0.0
    energy = np.zeros(len(r))
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    if np.any(mask_a):
        r_a = r[mask_a]
        energy[mask_a] = A * (r_a ** (-m) - B) * np.exp(c / (r_a - a))
    if np.any(mask_b):
        r_b = r[mask_b]
        energy[mask_b] = B * np.exp(d / (r_b - b))
    return float(np.sum(energy))

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    grad = np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    r2 = r2[i_idx, j_idx]
    r = np.sqrt(r2)
    rij = diff[i_idx, j_idx]
    valid = r > 1e-12
    r = r[valid]
    rij = rij[valid]
    i_valid = i_idx[valid]
    j_valid = j_idx[valid]
    if len(r) == 0:
        return grad
    dV_dr = np.zeros(len(r))
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    if np.any(mask_a):
        r_a = r[mask_a]
        exp_term = np.exp(c / (r_a - a))
        dV_dr[mask_a] = A * (-m * r_a ** (-m - 1)) * exp_term + A * (r_a ** (-m) - B) * exp_term * (-c / (r_a - a) ** 2)
    if np.any(mask_b):
        r_b = r[mask_b]
        dV_dr[mask_b] = B * np.exp(d / (r_b - b)) * (-d / (r_b - b) ** 2)
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    for k in range(len(forces)):
        i = i_valid[k]
        j = j_valid[k]
        grad[i] += forces[k]
        grad[j] -= forces[k]
    return grad

def _relax_structure(pos, max_iter=300):
    """Relax a structure using L-BFGS-B with gradient."""
    n = len(pos)
    x0 = pos.flatten()
    def fun(x):
        p = x.reshape(n, 3)
        return _dzugutov_energy(p)
    def jac(x):
        p = x.reshape(n, 3)
        return _dzugutov_gradient(p).flatten()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x.reshape(n, 3), res.fun
    except Exception:
        return pos, _dzugutov_energy(pos)

def _perturb_and_relax(pos, n_perturb=5, sigma=0.3, max_iter=200):
    """Perturb some atoms and re-relax."""
    n = len(pos)
    pos_pert = pos.copy()
    idx = np.random.choice(n, size=min(n_perturb, n), replace=False)
    pos_pert[idx] += np.random.randn(len(idx), 3) * sigma
    return _relax_structure(pos_pert, max_iter=max_iter)

def _get_icosahedron_vertices(radius=1.0):
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

def _get_icosahedron_edges(verts):
    """Get the 30 edge midpoint directions of an icosahedron."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    edge_dirs = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_dirs.append(mid)
    return np.array(edge_dirs)

def _get_icosahedron_face_centers(verts):
    """Get the 20 face center directions of an icosahedron."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    adj = {i: [] for i in range(n_verts)}
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                adj[i].append(j)
                adj[j].append(i)
    faces = []
    for i in range(n_verts):
        for j in adj[i]:
            for k in adj[j]:
                if k > j and k in adj[i]:
                    face = (i, j, k)
                    if face not in faces:
                        faces.append(face)
    face_dirs = []
    for face in faces:
        centroid = np.mean(verts_unit[list(face)], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        face_dirs.append(centroid)
    return np.array(face_dirs)

def _build_mackay(n_target, r1=1.5, r2v=2.6, r2e=2.6):
    """Build Mackay icosahedron and truncate/expand to n_target."""
    # Full Mackay 55: 1 + 12 + 12 + 30
    points = [np.zeros(3)]
    v1 = _get_icosahedron_vertices(r1)
    points.extend(list(v1))
    v1_unit = v1 / r1
    v2 = v1_unit * r2v
    points.extend(list(v2))
    e_dirs = _get_icosahedron_edges(_get_icosahedron_vertices(1.0))
    e2 = e_dirs * r2e
    points.extend(list(e2))
    # That's 55 points. If we need fewer, remove outermost.
    pos = np.array(points)
    if len(pos) == n_target:
        return pos
    elif len(pos) > n_target:
        # Remove from outermost shell first
        com = np.mean(pos, axis=0)
        dists = np.linalg.norm(pos - com, axis=1)
        idx_sorted = np.argsort(dists)[::-1]
        keep_mask = np.ones(len(pos), dtype=bool)
        keep_mask[idx_sorted[:len(pos) - n_target]] = False
        return pos[keep_mask]
    else:
        # Need more atoms - add at face centers
        extra = n_target - len(pos)
        f_dirs = _get_icosahedron_face_centers(_get_icosahedron_vertices(1.0))
        max_r = np.max(np.linalg.norm(pos, axis=1))
        r3 = max_r + 0.5
        # Pick extra face directions
        idx = np.random.choice(len(f_dirs), size=extra, replace=False)
        extra_pos = f_dirs[idx] * r3
        return np.vstack([pos, extra_pos])

def _build_decahedral(n_target, r1=1.5, r2=2.5):
    """Build a decahedral (5-fold symmetric) cluster."""
    # Decahedral: 5 tetrahedra sharing a common edge
    # Simplified: use icosahedron vertices but with decahedral distortion
    # For simplicity, use a slightly distorted icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * r1
    points = [np.zeros(3)]  # center
    points.extend(list(verts))  # 12 shell 1
    # Add shell 2: edge midpoints
    verts_unit = verts / r1
    min_dist = float('inf')
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    edge_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_dirs.append(mid)
    edge_pos = np.array(edge_dirs) * r2
    points.extend(list(edge_pos))
    pos = np.array(points)
    if len(pos) == n_target:
        return pos
    elif len(pos) > n_target:
        com = np.mean(pos, axis=0)
        dists = np.linalg.norm(pos - com, axis=1)
        idx_sorted = np.argsort(dists)[::-1]
        keep_mask = np.ones(len(pos), dtype=bool)
        keep_mask[idx_sorted[:len(pos) - n_target]] = False
        return pos[keep_mask]
    else:
        extra = n_target - len(pos)
        f_dirs = _get_icosahedron_face_centers(_get_icosahedron_vertices(1.0))
        max_r = np.max(np.linalg.norm(pos, axis=1))
        idx = np.random.choice(len(f_dirs), size=extra, replace=False)
        extra_pos = f_dirs[idx] * (max_r + 0.5)
        return np.vstack([pos, extra_pos])

def _build_tetrahedral(n_target, bond=1.5):
    """Build a polytetrahedral cluster by adding tetrahedra."""
    # Start with a tetrahedron (4 atoms)
    phi = (1 + np.sqrt(5)) / 2
    tet = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float) * bond / 2.0
    points = [tet[0].copy()]
    # Add atoms one by one, placing each at a tetrahedral site relative to existing atoms
    rng = np.random.RandomState(42)
    while len(points) < n_target:
        # Pick 3 existing atoms that form a roughly equilateral triangle
        # and place a new atom at the tetrahedral position
        n_pts = len(points)
        if n_pts < 3:
            # Add next vertex of initial tetrahedron
            if len(points) == 1:
                points.append(tet[1].copy())
            elif len(points) == 2:
                points.append(tet[2].copy())
            else:
                points.append(tet[3].copy())
            continue
        # Find a good triangle
        best_score = -1
        best_idx = None
        for i in range(min(n_pts, 10)):
            for j in range(i + 1, min(n_pts, 10)):
                for k in range(j + 1, min(n_pts, 10)):
                    d1 = np.linalg.norm(points[i] - points[j])
                    d2 = np.linalg.norm(points[j] - points[k])
                    d3 = np.linalg.norm(points[i] - points[k])
                    # Want roughly equilateral
                    score = 1.0 / (1.0 + (d1 - d2)**2 + (d2 - d3)**2 + (d1 - d3)**2)
                    if score > best_score:
                        best_score = score
                        best_idx = (i, j, k)
        if best_idx is None:
            # Just add a random atom near the cluster
            com = np.mean(points, axis=0)
            new_pos = com + np.random.randn(3) * 0.5
            points.append(new_pos)
            continue
        i, j, k = best_idx
        # Tetrahedral position: centroid + offset
        centroid = np.mean([points[i], points[j], points[k]], axis=0)
        # Direction from centroid to opposite vertex
        # For a regular tetrahedron, the 4th vertex is at centroid + h * normal
        # where h = bond * sqrt(2/3)
        v1 = np.array(points[j]) - np.array(points[i])
        v2 = np.array(points[k]) - np.array(points[i])
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-10:
            points.append(centroid + np.random.randn(3) * 0.3)
            continue
        normal = normal / np.linalg.norm(normal)
        h = bond * np.sqrt(2.0 / 3.0)
        # Two possible positions
        pos1 = centroid + normal * h
        pos2 = centroid - normal * h
        # Choose the one that's farther from existing atoms
        min_d1 = min(np.linalg.norm(pos1 - p) for p in points)
        min_d2 = min(np.linalg.norm(pos2 - p) for p in points)
        if min_d1 > min_d2:
            points.append(pos1)
        else:
            points.append(pos2)
    return np.array(points)

def _build_random_cluster(n_target, radius=3.0):
    """Build a random cluster within a sphere."""
    pos = np.random.randn(n_target, 3)
    norms = np.linalg.norm(pos, axis=1)
    pos = pos / norms[:, np.newaxis] * radius * np.random.uniform(0.5, 1.0, size=(n_target, 1))
    return pos

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    
    if _time_left() < 5:
        pos = np.random.randn(n, 3) * 0.5
        return pos

    best_energy = float('inf')
    best_pos = None
    
    candidates = []
    
    # Generate diverse structural families
    # 1. Mackay icosahedron variants
    for r1, r2v, r2e in [(1.4, 2.4, 2.4), (1.5, 2.6, 2.6), (1.6, 2.8, 2.8), (1.3, 2.3, 2.3)]:
        if _time_left() < 10:
            break
        try:
            pos = _build_mackay(n, r1, r2v, r2e)
            candidates.append(pos)
        except Exception:
            pass
    
    # 2. Decahedral variants
    for r1, r2 in [(1.4, 2.4), (1.5, 2.6), (1.6, 2.8)]:
        if _time_left() < 10:
            break
        try:
            pos = _build_decahedral(n, r1, r2)
            candidates.append(pos)
        except Exception:
            pass
    
    # 3. Polytetrahedral
    if _time_left() > 20:
        for bond in [1.4, 1.5, 1.6]:
            try:
                pos = _build_tetrahedral(n, bond)
                candidates.append(pos)
            except Exception:
                pass
    
    # 4. Random clusters
    if _time_left() > 15:
        for radius in [2.5, 3.0, 3.5]:
            pos = _build_random_cluster(n, radius)
            candidates.append(pos)
    
    if not candidates:
        pos = np.random.randn(n, 3) * 1.0
        candidates.append(pos)
    
    # Score and sort candidates by initial energy
    scored = []
    for pos in candidates:
        try:
            e = _dzugutov_energy(pos)
            scored.append((e, pos))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0])
    
    # Relax top candidates
    max_relax = min(len(scored), 8)
    for e_init, pos in scored[:max_relax]:
        if _time_left() < 15:
            break
        try:
            pos_rel, e_rel = _relax_structure(pos, max_iter=300)
            if e_rel < best_energy:
                best_energy = e_rel
                best_pos = pos_rel.copy()
        except Exception:
            continue
    
    # If time remains, do perturbation on the best
    if best_pos is not None and _time_left() > 20:
        for _ in range(3):
            if _time_left() < 15:
                break
            try:
                pos_pert, e_pert = _perturb_and_relax(best_pos, n_perturb=5, sigma=0.3, max_iter=200)
                if e_pert < best_energy:
                    best_energy = e_pert
                    best_pos = pos_pert.copy()
            except Exception:
                break
    
    if best_pos is None:
        if scored:
            best_pos = scored[0][1]
        else:
            best_pos = np.random.randn(n, 3) * 0.5
    
    return best_pos

# EVOLVE-BLOCK-END
