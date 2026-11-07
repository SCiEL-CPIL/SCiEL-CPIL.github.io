"""Sutton-Chen transition-metal cluster global minima solver."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Constants for the task
MAX_TIME = 290.0  # Safety margin before 300s limit
MIN_DIST = 0.35

def _generate_mackay(n):
    """
    Generate positions for a Mackay icosahedron with n atoms.
    Returns None if n is not a Mackay number or if construction fails.
    """
    # Mackay numbers: 1, 12, 42, 120, 292...
    # We approximate for arbitrary n by taking the first n atoms of a larger Mackay
    # or by filling shells.
    
    # Shells of a Mackay icosahedron:
    # Shell 0: 1 atom (center)
    # Shell 1: 12 atoms (vertices of icosahedron)
    # Shell 2: 30 atoms (midpoints of edges of icosahedron) -> Total 42
    # Shell 3: 60 atoms (face centers) -> Total 102? No, standard Mackay 120 has 60 more.
    # Actually, standard Mackay 120 consists of 5 interpenetrating icosahedra.
    # Let's construct the first few shells explicitly.
    
    phi = (1 + 5**0.5) / 2.0
    inv_phi = 1.0 / phi
    
    # Vertices of a regular icosahedron (radius 1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    verts /= np.linalg.norm(verts[0])
    
    # Edge midpoints (Shell 2)
    # An icosahedron has 30 edges.
    # We can find them by checking distances between vertices.
    # Distance between adjacent vertices is 2 * sin(pi/5) * R?
    # Let's just compute all pairs and pick the 30 shortest.
    dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    # Sort distances
    flat_dists = dists.flatten()
    flat_dists.sort()
    min_dist_edge = flat_dists[0]
    # Find edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i, j], min_dist_edge, atol=1e-5):
                edges.append((i, j))
    
    mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    mids /= np.linalg.norm(mids[0]) # Normalize to same radius as shell 1
    
    # Face centers (Shell 3)
    # An icosahedron has 20 faces.
    # We can find them by identifying triangles of vertices that are all connected.
    # Or simpler: use the known geometry.
    # For now, let's stick to Shells 0, 1, 2 (1 + 12 + 30 = 43 atoms approx)
    # Wait, 1+12+30 = 43. The 42nd atom is usually missing one vertex or something?
    # Actually, standard Mackay 42 is 1 + 12 + 29? No.
    # Let's just generate a large set of candidate points from these shells
    # and pick the first n.
    
    # Shell 3: Face centers.
    # Find 20 faces.
    faces = []
    # A face is a triangle of vertices where all 3 pairs are edges.
    edge_set = set(edges)
    for i in range(12):
        for j in range(i+1, 12):
            if (i,j) in edge_set:
                for k in range(j+1, 12):
                    if (j,k) in edge_set and (i,k) in edge_set:
                        faces.append((i,j,k))
    
    face_centers = np.array([(verts[i] + verts[j] + verts[k]) / 3.0 for i,j,k in faces])
    face_centers /= np.linalg.norm(face_centers[0])
    
    # Combine all candidate points
    # Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Scale factors for shells
    # The distance from center to vertex is R1.
    # Distance from center to edge midpoint is R2.
    # Distance from center to face center is R3.
    # In a Mackay icosahedron, these are specific ratios.
    # Let's assume the "unit" is the nearest neighbor distance in the relaxed cluster.
    # For a crude initialization, we can just use the normalized vectors scaled by a base radius.
    
    base_radius = 1.0
    points = np.vstack([
        center,
        verts * base_radius,
        mids * base_radius * 0.95, # Slightly smaller to avoid overlap issues initially
        face_centers * base_radius * 0.85
    ])
    
    # If we need more than 1+12+30+20 = 63 points, we can generate a second shell (Mackay 120 level)
    # But for n up to 80, 63 is close. We can add more points on the surface.
    # Let's just take the first n points.
    if n <= len(points):
        return points[:n]
    else:
        # If n is larger, we might need to expand. 
        # For simplicity in this heuristic, if n > 63, we return None and rely on FCC/HCP or random.
        # Or we can just repeat the process with a larger radius? 
        # Let's try to generate a second layer of vertices scaled up.
        verts2 = verts * base_radius * 2.0
        mids2 = mids * base_radius * 1.9
        face_centers2 = face_centers * base_radius * 1.7
        points = np.vstack([points, verts2, mids2, face_centers2])
        if n <= len(points):
            return points[:n]
        return None

def _generate_fcc(n):
    """
    Generate positions for a truncated FCC cluster.
    """
    # Generate FCC lattice points within a sphere
    # FCC lattice vectors:
    # (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Unit cell edge a. Nearest neighbor distance d = a * sqrt(2) / 2 = a / sqrt(2).
    # We want d ~ 1.0, so a = sqrt(2).
    a = np.sqrt(2)
    
    # Estimate radius for n atoms.
    # Density of FCC is 4 atoms / a^3.
    # Volume V = n * a^3 / 4.
    # R = (3V/4pi)^(1/3)
    R = (3.0 * n * a**3 / (4.0 * 4.0 * np.pi)) ** (1.0/3.0)
    
    # Generate grid
    # Max index needed: R / (a/2) roughly.
    max_idx = int(np.ceil(R / (a/2.0))) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # 4 atoms per unit cell
                for dx, dy, dz in [(0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    if len(points) < n:
        # Increase radius slightly
        R *= 1.1
        max_idx = int(np.ceil(R / (a/2.0))) + 2
        points = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                for k in range(-max_idx, max_idx + 1):
                    for dx, dy, dz in [(0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                        p = np.array([i + dx, j + dy, k + dz]) * a
                        if np.linalg.norm(p) <= R:
                            points.append(p)
        
    if len(points) < n:
        return None
        
    # Sort by distance from center to ensure compactness
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    return points[sorted_indices[:n]]

def _generate_hcp(n):
    """
    Generate positions for a truncated HCP cluster.
    """
    # HCP is similar to FCC but with ABAB stacking instead of ABC.
    # For initialization, a truncated HCP sphere is hard to generate programmatically quickly.
    # We can approximate by taking FCC and shearing? Or just use FCC and Random.
    # Let's skip explicit HCP generation for now and rely on FCC and Random, 
    # as the relaxation will find the local minimum which might be HCP-like.
    return None

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    best_energy = np.inf
    best_pos = None
    
    # 1. Try structured initializations
    candidates = []
    
    # Mackay
    mackay_pos = _generate_mackay(n)
    if mackay_pos is not None:
        candidates.append(("mackay", mackay_pos))
        
    # FCC
    fcc_pos = _generate_fcc(n)
    if fcc_pos is not None:
        candidates.append(("fcc", fcc_pos))
        
    # Random compact balls (fallback and for diversity)
    # Use a few random seeds
    rng = np.random.default_rng(42)
    for i in range(5):
        # Pack spheres roughly
        radius = 0.72 * n ** (1.0/3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check distance to existing points
                if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
        if len(points) == n:
            candidates.append((f"random_{i}", np.array(points)))
            
    # 2. Relax each candidate
    for name, pos in candidates:
        if time.time() - start_time > MAX_TIME - 5:
            break
            
        try:
            # Ensure no atoms are too close initially to avoid numerical issues in relax
            # The relax function should handle this, but let's be safe.
            energy, relaxed_pos = relax(pos, "sutton_chen", kind)
            
            if np.isfinite(energy) and energy < best_energy:
                # Check validity: no atoms closer than 0.35
                min_d = np.inf
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                        if d < min_d:
                            min_d = d
                if min_d >= MIN_DIST:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            pass

    # 3. If time remains, do some local search/perturbation on the best structure
    if best_pos is not None and time.time() - start_time < MAX_TIME - 10:
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        # Basin hopping lite: perturb and relax
        rng = np.random.default_rng(123)
        while time.time() - start_time < MAX_TIME - 5:
            # Perturb
            perturbation_size = 0.2
            perturbed = current_pos + rng.normal(0, perturbation_size, (n, 3))
            
            try:
                energy, new_pos = relax(perturbed, "sutton_chen", kind)
                if np.isfinite(energy):
                    # Check validity
                    min_d = np.inf
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.linalg.norm(new_pos[i] - new_pos[j])
                            if d < min_d:
                                min_d = d
                    if min_d >= MIN_DIST:
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = new_pos
                            current_pos = new_pos
                            current_energy = energy
                        # Even if not better, we might use it as a new center for perturbation
                        # if we are stuck, but for now just accept improvements.
            except Exception:
                pass

    if best_pos is None:
        # Fallback: return a simple linear chain or random if everything failed
        # This shouldn't happen, but better to return something valid.
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
