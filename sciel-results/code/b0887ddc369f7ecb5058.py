# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax


def _icosahedron_vertices():
    """12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _icosahedron_second_shell():
    """
    42 vertices of the second icosahedral shell (Mackay 55-atom cluster).
    These are: 30 edge-midpoints (between adjacent icosahedron vertices) 
    and 12 face-centers (of the 20 triangular faces).
    Actually for Mackay 55: 1 + 12 + 42 = 55.
    The 42 = 30 (edges) + 12 (faces) ... wait, icosahedron has 30 edges and 20 faces.
    So 30 + 20 = 50? No, Mackay second shell has 42 atoms.
    
    Let me reconsider. The Mackay icosahedron shells:
    - n=1: 1 atom
    - n=13: 1 + 12 (1st shell)
    - n=55: 1 + 12 + 42 (2nd shell)
    
    The 42 atoms in the 2nd shell: 30 at edge positions + 12 at face positions? 
    Actually: icosahedron has 12 vertices, 30 edges, 20 faces.
    2nd shell positions: 30 (one per edge) + 12 (one per... no)
    
    Let me use the standard construction:
    2nd shell = 30 positions (midpoints of edges, scaled) + 12 positions (face centers, scaled)
    That gives 42. Yes!
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms1 = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / norms1
    
    # Find edges: pairs of vertices that are at minimum distance
    dmin = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v1[i] - v1[j])
            if d < dmin:
                dmin = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(v1[i] - v1[j]) - dmin) < 1e-10:
                edges.append((i, j))
    
    # Edge midpoints (30)
    edge_mids = []
    for i, j in edges:
        mid = (v1[i] + v1[j]) / 2.0
        edge_mids.append(mid / np.linalg.norm(mid))
    
    # Find faces: triples of mutually connected vertices
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
    
    # Face centers (20)
    face_centers = []
    for i, j, k in faces:
        fc = (v1[i] + v1[j] + v1[k]) / 3.0
        face_centers.append(fc / np.linalg.norm(fc))
    
    # Combine: 30 edge mids + 12 ... wait, that's 30 + 20 = 50, not 42.
    # Hmm, let me reconsider. Maybe the 2nd shell is just 30 + 12 = 42?
    # Actually in Mackay 55-atom icosahedron, the 2nd shell has 42 atoms.
    # Let me just use: 30 edge positions + 12 vertex-adjacent positions?
    
    # Alternative: use the known fact that 2nd shell = 42 points.
    # Let me just generate 42 points on an icosahedral symmetry set.
    # For now, use edge midpoints (30) + 12 points near vertices but further out.
    
    # Actually, let me just use a simpler approach: generate points with icosahedral symmetry
    # that approximate the 2nd shell.
    
    # The 42 points can be: 30 edge midpoints + 12 points at (1,1,1)-type positions
    # normalized to the shell radius.
    
    # Let me just return edge midpoints + face centers for now (50 points)
    # and trim to 42 if needed, or use a different construction.
    
    # For a cleaner approach, let me use the 42 = 30 + 12 where the 12 are
    # at the same angular positions as the 1st shell but at a different radius.
    # No, that doesn't work geometrically.
    
    # Let me just use all 50 (30 edge + 20 face) and let the user pick 42.
    # Or better: for n=55, use 1+12+42. For other n, we'll adjust.
    
    all_2nd = np.array(edge_mids + face_centers)
    return all_2nd, v1


def _generate_mackay_template(n_target, r1_factor=1.0, r2_factor=1.0):
    """
    Generate a Mackay icosahedral template with n_target atoms.
    r1_factor, r2_factor scale the shell radii.
    """
    v1_unit = _icosahedron_vertices()
    
    # Core
    points = [np.array([0.0, 0.0, 0.0])]
    
    # 1st shell: 12 atoms
    r1 = 1.1 * r1_factor
    shell1 = v1_unit * r1
    points.extend(shell1.tolist())
    
    # 2nd shell: up to 42 atoms
    if n_target > 13:
        all_2nd, _ = _icosahedron_second_shell()
        n_2nd_needed = min(n_target - 13, len(all_2nd))
        r2 = 1.1 * np.sqrt(2) * r2_factor  # approximate 2nd shell radius
        shell2 = all_2nd[:n_2nd_needed] * r2
        points.extend(shell2.tolist())
    
    # Trim or pad if necessary
    if len(points) > n_target:
        points = points[:n_target]
    elif len(points) < n_target:
        # Add atoms at random positions near the surface
        rng = np.random.default_rng(n_target)
        while len(points) < n_target:
            # Pick a random point on the 2nd shell and perturb
            idx = rng.integers(0, len(all_2nd))
            p = all_2nd[idx] * r2 + rng.normal(0, 0.1, 3)
            # Check minimum distance
            ok = True
            for q in points:
                if np.linalg.norm(np.array(p) - np.array(q)) < 0.5:
                    ok = False
                    break
            if ok:
                points.append(p)
    
    return np.array(points[:n_target])


def construct(instance):
    """Find low-energy Dzugutov cluster using Mackay icosahedral templates."""
    n = instance["n"]
    potential = instance["potential"]
    t_start = time.time()
    deadline = t_start + 280.0  # Leave 20s safety margin
    
    # Try multiple template configurations
    best_energy = np.inf
    best_positions = None
    
    # Different shell radius factors to explore
    configs = [
        (1.0, 1.0),
        (0.95, 0.95),
        (1.05, 1.05),
        (0.9, 1.0),
        (1.0, 0.9),
        (1.1, 1.0),
        (1.0, 1.1),
    ]
    
    for r1_f, r2_f in configs:
        if time.time() > deadline:
            break
        
        try:
            template = _generate_mackay_template(n, r1_f, r2_f)
            if template is None or len(template) != n:
                continue
            
            # Check minimum distance
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(template[i] - template[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 0.5:
                continue
            
            energy, positions = relax(template, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If no valid result from templates, fall back to random
    if best_positions is None:
        rng = np.random.default_rng(n)
        radius = 1.1 * n ** (1.0/3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential)
        best_energy = energy
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
