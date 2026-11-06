# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _remaining_time(deadline):
    return deadline - time.time()


def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit radius)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _get_icosahedron_edges():
    """Return list of edge index pairs for the icosahedron."""
    v = _get_icosahedron_vertices()
    dmin = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if d < dmin:
                dmin = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(v[i] - v[j]) - dmin) < 1e-10:
                edges.append((i, j))
    return edges


def _get_icosahedron_faces():
    """Return list of face index triples for the icosahedron."""
    edges = _get_icosahedron_edges()
    adj = [[False]*12 for _ in range(12)]
    for i, j in edges:
        adj[i][j] = True
        adj[j][i] = True
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if adj[i][j]:
                for k in range(j+1, 12):
                    if adj[i][k] and adj[j][k]:
                        faces.append((i, j, k))
    return faces


def _relax_fragment(positions, potential, max_time=5.0):
    """Relax a small fragment, return (energy, positions)."""
    t0 = time.time()
    try:
        energy, pos = relax(positions, potential)
        return energy, pos
    except Exception:
        return np.inf, positions


def _build_icosahedron_fragment(center, radius, seed=None):
    """Build a relaxed 13-atom icosahedral fragment at given center and radius."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    v = _get_icosahedron_vertices()
    # Add slight random perturbation to break symmetry
    v = v + rng.normal(0, 0.02, v.shape)
    
    positions = np.vstack([
        np.array(center, dtype=float),
        v * radius
    ])
    
    energy, positions = _relax_fragment(positions, 'dzugutov')
    return energy, positions


def _build_decahedron_fragment(center, radius, seed=None):
    """Build a 14-atom decahedral fragment (13 icosahedron + 1 defect)."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    v = _get_icosahedron_vertices()
    v = v + rng.normal(0, 0.03, v.shape)
    
    # Add a 14th atom at a slightly perturbed position
    extra_dir = v[rng.integers(0, 12)] + rng.normal(0, 0.1, 3)
    extra_dir = extra_dir / np.linalg.norm(extra_dir)
    
    positions = np.vstack([
        np.array(center, dtype=float),
        v * radius,
        extra_dir * radius * 1.05
    ])
    
    energy, positions = _relax_fragment(positions, 'dzugutov')
    return energy, positions


def _random_rotation_matrix(rng):
    """Generate a random 3x3 rotation matrix."""
    # Use QR decomposition of a random matrix
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    d = np.diag(R)
    ph = np.sign(d)
    Q = Q * ph
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _place_fragments(n_target, deadline, seed=42):
    """
    Fragment-merger strategy: build small icosahedral fragments and place them
    to form a cluster of n_target atoms.
    
    Returns list of (fragment_positions, fragment_energy) tuples and their placements.
    """
    rng = np.random.default_rng(seed)
    
    # Determine fragment composition
    # Use 13-atom icosahedra as primary building blocks
    # For n atoms, use floor(n/13) full icosahedra + remainder
    n_full = n_target // 13
    remainder = n_target % 13
    
    if n_full == 0:
        n_full = 1
        remainder = n_target - 13
    
    # Place fragments with some overlap/interpenetration to form a connected cluster
    # Strategy: place fragments at vertices of a smaller polyhedron
    # or in a linear/cubic arrangement
    
    fragments = []
    fragment_centers = []
    
    # Generate fragment placement positions
    if n_full == 1:
        centers = [np.array([0.0, 0.0, 0.0])]
    elif n_full == 2:
        # Two fragments side by side
        sep = 2.2
        centers = [
            np.array([-sep/2, 0, 0]),
            np.array([sep/2, 0, 0])
        ]
    elif n_full == 3:
        # Three fragments in a triangle
        sep = 2.5
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        centers = [sep * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    else:
        # More fragments: use a cubic-ish arrangement
        sep = 2.5
        centers = []
        k = 0
        while len(centers) < n_full:
            # Place in a 3D grid pattern
            x = (k % 3) - 1
            y = ((k // 3) % 3) - 1
            z = (k // 9) - 1
            if abs(x) + abs(y) + abs(z) <= 2:  # Keep it compact
                centers.append(sep * np.array([x, y, z]))
            k += 1
        centers = centers[:n_full]
    
    # Build and place each fragment
    for i, center in enumerate(centers):
        if time.time() > deadline:
            break
        
        # Random rotation for each fragment
        R = _random_rotation_matrix(rng)
        radius = 1.05 + rng.uniform(-0.05, 0.05)
        
        v = _get_icosahedron_vertices()
        v_rot = (R @ v.T).T
        v_rot = v_rot + rng.normal(0, 0.02, v_rot.shape)
        
        frag_pos = np.vstack([
            center,
            center + v_rot * radius
        ])
        
        energy, frag_pos = _relax_fragment(frag_pos, 'dzugutov')
        fragments.append((frag_pos, energy))
        fragment_centers.append(center)
    
    # Handle remainder atoms
    if remainder > 0 and time.time() < deadline:
        # Add remainder atoms near the surface of existing fragments
        last_frag = fragments[-1][0] if fragments else np.array([[0,0,0]])
        v = _get_icosahedron_vertices()
        R = _random_rotation_matrix(rng)
        v_rot = (R @ v.T).T
        
        # Pick remainder vertices from the icosahedron
        idx = rng.choice(12, size=min(remainder, 12), replace=False)
        rem_pos = last_frag[0] + v_rot[idx] * 1.2
        rem_pos = rem_pos + rng.normal(0, 0.1, rem_pos.shape)
        
        # Relax the remainder fragment
        energy_rem, rem_pos = _relax_fragment(rem_pos, 'dzugutov')
        fragments.append((rem_pos, energy_rem))
    
    return fragments, fragment_centers


def _merge_fragments(fragments, deadline):
    """
    Coarse-grained annealing: minimize inter-fragment repulsion by
    translating/rotating fragments to reduce overlap.
    """
    if len(fragments) <= 1:
        all_pos = np.vstack([f[0] for f in fragments])
        return all_pos
    
    # Compute fragment centroids and sizes
    centroids = [f[0].mean(axis=0) for f in fragments]
    sizes = [len(f[0]) for f in fragments]
    
    # Coarse-grained energy: sum of pairwise repulsion between fragment centers
    def cg_energy(transforms):
        """transforms: list of (rotation_matrix, translation_vector)"""
        e = 0.0
        new_centroids = []
        for i, (R, t) in enumerate(transforms):
            c = R @ centroids[i] + t
            new_centroids.append(c)
        
        # Repulsion between fragment centers
        for i in range(len(new_centroids)):
            for j in range(i+1, len(new_centroids)):
                d = np.linalg.norm(new_centroids[i] - new_centroids[j])
                if d < 3.0:
                    e += 50.0 * (3.0 - d)**2
        
        return e
    
    # Start with identity transforms
    transforms = [(np.eye(3), np.zeros(3)) for _ in fragments]
    
    # Simple gradient-free optimization of fragment positions
    rng = np.random.default_rng(123)
    best_e = cg_energy(transforms)
    best_transforms = [t.copy() if hasattr(t, 'copy') else (t[0].copy(), t[1].copy()) for t in transforms]
    
    n_iter = min(50, int(_remaining_time(deadline) * 10))
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Perturb one fragment's position
        i = rng.integers(0, len(fragments))
        R, t = transforms[i]
        t_new = t + rng.normal(0, 0.3, 3)
        
        transforms[i] = (R, t_new)
        e = cg_energy(transforms)
        
        if e < best_e:
            best_e = e
            best_transforms = [t.copy() if hasattr(t, 'copy') else (t[0].copy(), t[1].copy()) for t in transforms]
        else:
            transforms[i] = (R, t)
    
    # Apply best transforms
    all_pos = []
    for i, (frag_pos, _) in enumerate(fragments):
        R, t = best_transforms[i]
        # Center fragment at origin, rotate, translate
        centered = frag_pos - centroids[i]
        transformed = (R @ centered.T).T + t
        all_pos.append(transformed)
    
    return np.vstack(all_pos)


def _ensure_min_distance(positions, min_dist=0.5):
    """Ensure all pairwise distances are at least min_dist."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                # Push apart
                if d > 1e-10:
                    direction = (positions[i] - positions[j]) / d
                    push = (min_dist - d) / 2.0 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
                else:
                    positions[i] += np.array([0.01, 0, 0])
    return positions


def _basin_hopping(positions, potential, deadline, max_iter=30):
    """Simple basin hopping: perturb and relax."""
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    best_energy = np.inf
    best_pos = positions.copy()
    
    t0 = time.time()
    energy, best_pos = relax(positions, potential)
    best_energy = energy
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if _remaining_time(deadline) < 5.0:
            break
        
        # Perturb
        scale = 0.3 + 0.2 * (it / max_iter)
        perturbed = best_pos + rng.normal(0, scale, best_pos.shape)
        perturbed = _ensure_min_distance(perturbed, 0.5)
        
        # Relax
        try:
            e, pos = relax(perturbed, potential)
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except Exception:
            continue
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using fragment-merger strategy."""
    n = instance["n"]
    potential = instance["potential"]
    t_start = time.time()
    deadline = t_start + 275.0  # 55s safety margin
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Fragment-merger with icosahedral building blocks
    for seed in [42, 123, 777]:
        if _remaining_time(deadline) < 10.0:
            break
        
        try:
            fragments, centers = _place_fragments(n, deadline, seed=seed)
            if not fragments:
                continue
            
            merged = _merge_fragments(fragments, deadline)
            merged = _ensure_min_distance(merged, 0.5)
            
            if len(merged) != n:
                # Trim or pad
                if len(merged) > n:
                    merged = merged[:n]
                else:
                    # Add missing atoms
                    rng = np.random.default_rng(seed)
                    while len(merged) < n:
                        p = merged[-1] + rng.normal(0, 0.5, 3)
                        if all(np.linalg.norm(p - q) >= 0.5 for q in merged):
                            merged = np.vstack([merged, p])
            
            energy, positions = _basin_hopping(merged, potential, deadline)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Strategy 2: Single large icosahedral template (fallback)
    if _remaining_time(deadline) > 10.0:
        try:
            v = _get_icosahedron_vertices()
            rng = np.random.default_rng(n)
            
            # Build a larger icosahedral-like structure
            points = [np.array([0.0, 0.0, 0.0])]
            r1 = 1.1
            shell1 = v * r1
            points.extend(shell1.tolist())
            
            # Add more shells
            if n > 13:
                edges = _get_icosahedron_edges()
                faces = _get_icosahedron_faces()
                
                edge_mids = []
                for i, j in edges:
                    mid = (v[i] + v[j]) / 2.0
                    edge_mids.append(mid / np.linalg.norm(mid))
                
                face_centers = []
                for i, j, k in faces:
                    fc = (v[i] + v[j] + v[k]) / 3.0
                    face_centers.append(fc / np.linalg.norm(fc))
                
                all_2nd = np.array(edge_mids + face_centers)
                n_2nd = min(n - 13, len(all_2nd))
                r2 = 1.1 * np.sqrt(2)
                shell2 = all_2nd[:n_2nd] * r2
                points.extend(shell2.tolist())
            
            template = np.array(points[:n])
            template = _ensure_min_distance(template, 0.5)
            
            energy, positions = _basin_hopping(template, potential, deadline)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Fallback: random compact structure
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.1 * n ** (1.0/3.0) * 0.7
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        if len(points) >= n:
            points = np.array(points[:n])
            energy, positions = relax(points, potential)
            best_energy = energy
            best_positions = positions
    
    if best_positions is None:
        # Last resort
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = np.array([
                (i % 5) * 0.8 - 1.6,
                (i // 5 % 5) * 0.8 - 1.6,
                (i // 25) * 0.8 - 0.8
            ])
        best_positions = _ensure_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
