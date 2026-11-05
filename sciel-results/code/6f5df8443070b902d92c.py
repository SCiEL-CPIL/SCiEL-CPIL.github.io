"""Dzugutov cluster global minima search using explicit Mackay icosahedron construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

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
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    # Only compute upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    r2 = r2[i_idx, j_idx]
    r = np.sqrt(r2)
    
    # Filter out very small distances
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
    
    # Compute all pairwise vectors and distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    r2 = r2[i_idx, j_idx]
    r = np.sqrt(r2)
    rij = diff[i_idx, j_idx]
    
    # Filter out very small distances
    valid = r > 1e-12
    r = r[valid]
    rij = rij[valid]
    i_valid = i_idx[valid]
    j_valid = j_idx[valid]
    
    if len(r) == 0:
        return grad
    
    # Compute dV/dr for each pair
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
    
    # Compute force factors: -dV_dr / r
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    
    # Accumulate gradients using np.add.at
    for k in range(len(forces)):
        i = i_valid[k]
        j = j_valid[k]
        grad[i] += forces[k]
        grad[j] -= forces[k]
    
    return grad

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
    """Get the 30 edge midpoints (directions) of an icosahedron given its vertices."""
    n_verts = len(verts)
    # Unit vectors
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # Find edges: pairs of vertices separated by the minimum distance
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
            elif abs(d - min_dist) < 1e-6:
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
    # For a regular icosahedron, face centers are in the directions of the 20 faces
    # We can compute them by finding triangles and taking their centroids
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # Find all faces (triangles) - edges connect vertices at minimum distance
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
            elif abs(d - min_dist) < 1e-6:
                min_dist = d
    
    # Build adjacency
    adj = {i: [] for i in range(n_verts)}
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                adj[i].append(j)
                adj[j].append(i)
    
    # Find faces: each face is a triangle where all three vertices are connected
    faces = []
    for i in range(n_verts):
        for j in adj[i]:
            for k in adj[j]:
                if k > j and k in adj[i]:
                    # Triangle i-j-k
                    face = (i, j, k)
                    if face not in faces:
                        faces.append(face)
    
    # Compute face center directions
    face_dirs = []
    for face in faces:
        centroid = np.mean(verts_unit[list(face)], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        face_dirs.append(centroid)
    
    return np.array(face_dirs)

def _build_mackay_55(r1, r2v, r2e):
    """
    Build Mackay 55 icosahedron.
    r1: radius of shell 1 (12 atoms)
    r2v: radius of shell 2 vertices (12 atoms, same directions as shell 1)
    r2e: radius of shell 2 edges (30 atoms, edge midpoint directions of shell 1)
    """
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 vertices
    v1 = _get_icosahedron_vertices(r1)
    points.extend([v for v in v1])
    
    # Shell 2: 12 vertices + 30 edge centers
    # Directions are same as v1 unit vectors
    v1_unit = v1 / r1
    v2 = v1_unit * r2v
    points.extend([v for v in v2])
    
    # Edge directions from unit icosahedron
    e_dirs = _get_icosahedron_edges(_get_icosahedron_vertices(1.0))
    e2 = e_dirs * r2e
    points.extend([e for e in e2])
    
    return np.array(points)

def _build_mackay_55_with_faces(r1, r2v, r2e, r3f):
    """
    Build extended Mackay icosahedron with face centers.
    r1: radius of shell 1 (12 atoms)
    r2v: radius of shell 2 vertices (12 atoms)
    r2e: radius of shell 2 edges (30 atoms)
    r3f: radius of shell 3 face centers (20 atoms)
    """
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 vertices
    v1 = _get_icosahedron_vertices(r1)
    points.extend([v for v in v1])
    
    # Shell 2: 12 vertices + 30 edge centers
    v1_unit = v1 / r1
    v2 = v1_unit * r2v
    points.extend([v for v in v2])
    
    e_dirs = _get_icosahedron_edges(_get_icosahedron_vertices(1.0))
    e2 = e_dirs * r2e
    points.extend([e for e in e2])
    
    # Shell 3: 20 face centers
    f_dirs = _get_icosahedron_face_centers(_get_icosahedron_vertices(1.0))
    f3 = f_dirs * r3f
    points.extend([f for f in f3])
    
    return np.array(points)

def _relax_structure(pos, max_iter=200):
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

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    
    # Check time
    if _time_left() < 5:
        # Emergency: return a simple cluster
        pos = np.random.randn(n, 3) * 0.5
        return pos

    best_energy = float('inf')
    best_pos = None
    
    # Strategy: 
    # 1. If n <= 55, use subsets of Mackay 55.
    # 2. If n > 55, use Mackay 55 + extra atoms.
    
    # Define radii guesses
    # Optimal bond length for Dzugutov is approx 1.5-1.8?
    # Let's try a few sets of radii.
    # Standard Mackay geometry ratios: 
    # If r1 = 1.0, r2v = sqrt(1+phi) ~ 1.7? r2e = sqrt(1+phi/2)?
    # Let's just use a range.
    radii_sets = [
        (1.5, 2.5, 2.5),
        (1.6, 2.6, 2.6),
        (1.7, 2.8, 2.8),
        (1.5, 2.4, 2.4),
        (1.8, 3.0, 3.0),
    ]
    
    # Generate candidates
    candidates = []
    
    if n <= 55:
        # Generate Mackay 55 for each radius set
        mackay_55s = []
        for r1, r2v, r2e in radii_sets:
            m55 = _build_mackay_55(r1, r2v, r2e)
            # Relax the 55-atom cluster first to get optimal geometry
            # This is expensive, so we might do it once or twice
            if _time_left() < 20:
                break
            try:
                pos55, e55 = _relax_structure(m55, max_iter=100)
                mackay_55s.append((pos55, e55))
            except Exception:
                pass
        
        # For each relaxed 55-atom cluster, create n-atom subsets
        for pos55, e55 in mackay_55s:
            if n == 55:
                candidates.append(pos55)
            else:
                # Create subset by removing outermost atoms
                # Calculate distances from centroid
                com = np.mean(pos55, axis=0)
                dists = np.linalg.norm(pos55 - com, axis=1)
                # Indices of atoms, sorted by distance (largest first)
                idx_sorted = np.argsort(dists)[::-1]
                # Remove (55-n) atoms
                remove_idx = idx_sorted[:55-n]
                keep_mask = np.ones(55, dtype=bool)
                keep_mask[remove_idx] = False
                pos_subset = pos55[keep_mask]
                candidates.append(pos_subset)
                
                # Also try removing specific types (e.g. only edge centers or only vertex centers)
                # This is heuristic, but might help escape bad local minima
                # We'll just rely on the random-ish selection from multiple radius sets
                
    else:
        # n > 55
        # Start with Mackay 55
        for r1, r2v, r2e in radii_sets[:3]: # Fewer sets for larger n
            if _time_left() < 20:
                break
            m55 = _build_mackay_55(r1, r2v, r2e)
            try:
                pos55, e55 = _relax_structure(m55, max_iter=100)
                # Add extra atoms
                # Place them at the face centers of the outermost shell?
                # Or just randomly on a sphere of radius r2e + 0.5
                extra_n = n - 55
                # Simple heuristic: add atoms near the face centers of the icosahedron
                # Get face normals of the icosahedron defined by shell 2 vertices
                # This is complex. Let's just add them in a ring or random shell.
                # Better: Take the direction of the 20 face centers of the icosahedron
                v1 = _get_icosahedron_vertices(1.0)
                # Compute face normals (cross product of edges)
                # We need the faces.
                # For simplicity, just scatter them on a sphere of radius pos55 max radius + 0.5
                max_r = np.max(np.linalg.norm(pos55, axis=1))
                extra_pos = np.random.randn(extra_n, 3)
                extra_pos *= (max_r + 0.5) / np.linalg.norm(extra_pos, axis=1)[:, np.newaxis]
                
                pos_n = np.vstack([pos55, extra_pos])
                candidates.append(pos_n)
            except Exception:
                pass

    # Fallback if no candidates generated
    if not candidates:
        pos = np.random.randn(n, 3) * 1.0
        candidates.append(pos)

    # Relax candidates
    # Prioritize candidates with lower initial energy if possible, or just do them all
    # To save time, we can sort candidates by initial energy
    scored_candidates = []
    for pos in candidates:
        try:
            e = _dzugutov_energy(pos)
            scored_candidates.append((e, pos))
        except Exception:
            continue
    
    # Sort by energy
    scored_candidates.sort(key=lambda x: x[0])
    
    # Relax the best few
    max_relaxations = 5
    for e_init, pos in scored_candidates[:max_relaxations]:
        if _time_left() < 10:
            break
        try:
            pos_rel, e_rel = _relax_structure(pos, max_iter=150)
            if e_rel < best_energy:
                best_energy = e_rel
                best_pos = pos_rel
        except Exception:
            continue

    if best_pos is None:
        # Return the best initial candidate
        if scored_candidates:
            best_pos = scored_candidates[0][1]
        else:
            best_pos = np.random.randn(n, 3) * 0.5
            
    return best_pos

# EVOLVE-BLOCK-END
