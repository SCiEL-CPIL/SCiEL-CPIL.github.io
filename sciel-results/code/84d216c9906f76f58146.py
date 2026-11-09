import numpy as np
from autoresearch_suite.chem_clusters import relax

def generate_mackay_icosahedron(edge_length):
    """
    Generates the coordinates of a 55-atom Mackay icosahedron.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices scaled to have edge length 1
    # Vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit edge length
    # Distance between [0,1,phi] and [0,1,-phi] is 2*phi.
    # Distance between [0,1,phi] and [1,phi,0] is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 0.382 + 2.618) = sqrt(4) = 2.
    # Wait, standard coords have edge length 2.
    # Let's verify.
    # v1 = [0, 1, phi], v2 = [1, phi, 0]
    # diff = [1, phi-1, -phi]
    # norm^2 = 1 + (phi-1)^2 + phi^2 = 1 + (phi^2 - 2phi + 1) + phi^2 = 2 + 2phi^2 - 2phi
    # phi^2 = phi + 1
    # norm^2 = 2 + 2(phi+1) - 2phi = 2 + 2phi + 2 - 2phi = 4.
    # So edge length is 2.
    verts = verts / 2.0 * edge_length

    # Center atom
    center = np.array([0, 0, 0])
    
    # Edge midpoints
    # An icosahedron has 30 edges.
    # We need to identify all pairs of vertices that are neighbors.
    # Two vertices are neighbors if their distance is edge_length.
    edge_midpoints = []
    n_verts = len(verts)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            dist = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(dist, edge_length):
                mid = (verts[i] + verts[j]) / 2.0
                edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)

    # Face centers
    # An icosahedron has 20 faces.
    # We need to identify all triplets of vertices that form a face.
    # A face is formed by 3 mutually connected vertices.
    # Alternatively, we can project the center to the plane of the face.
    # Or simpler: The 20 face centers of an icosahedron are the vertices of a dodecahedron.
    # The dodecahedron vertices can be generated from the icosahedron.
    # Or we can just find the 20 sets of 3 vertices that are mutually at distance edge_length.
    face_centers = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            if np.isclose(np.linalg.norm(verts[i] - verts[j]), edge_length):
                for k in range(j + 1, n_verts):
                    if (np.isclose(np.linalg.norm(verts[i] - verts[k]), edge_length) and
                        np.isclose(np.linalg.norm(verts[j] - verts[k]), edge_length)):
                        # i, j, k form a face
                        center = (verts[i] + verts[j] + verts[k]) / 3.0
                        # Check if this center is already in the list (to avoid duplicates)
                        # Actually, each face is unique.
                        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Combine all
    atoms = np.vstack([center[None, :], verts, edge_midpoints, face_centers])
    return atoms

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    # Ideal edge length for Dzugutov
    # The minimum of V(r) is around r ~ 1.05
    L = 1.05
    
    candidates = []
    
    # 1. Mackay Icosahedron (55 atoms)
    if n <= 55:
        icos = generate_mackay_icosahedron(L)
        # Remove atoms to reach n
        # Calculate distance from center for each atom
        dists = np.linalg.norm(icos, axis=1)
        # Sort by distance, remove furthest
        indices = np.argsort(dists)[::-1]
        selected_indices = indices[55-n:55] # Keep the closest ones
        # Wait, if n < 55, we keep the first n closest?
        # indices[0] is furthest. indices[-1] is closest.
        # We want to keep the closest n.
        # So we take the last n elements of the sorted indices (ascending distance)
        indices_asc = np.argsort(dists)
        selected_indices = indices_asc[:n]
        candidates.append(icos[selected_indices])
        
        # Also try removing specifically from the outer shell (face centers)
        # The face centers are the furthest.
        # If n is close to 55, this is good.
        
    # 2. Random Close Packing
    rng = np.random.default_rng(42)
    radius = L * n ** (1/3) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.9 * L for q in points):
                points.append(p)
                attempts = 0
            else:
                attempts += 1
    if len(points) == n:
        candidates.append(np.array(points))

    # 3. FCC Fragment
    # Generate FCC points in a sphere
    a = L / np.sqrt(2) # FCC nearest neighbor distance is a/sqrt(2)
    # So a = L * sqrt(2)
    a = L * np.sqrt(2)
    fcc_points = []
    max_coord = int(a * n ** (1/3) * 1.5)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC lattice points: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
                # Actually, simpler: generate all integer combinations and filter
                pass # Too slow/complex. Let's skip FCC or do a simple version.
                
    # Let's stick to Icosahedron and Random.
    # Add a "deformed" icosahedron if n is close to 55
    if 50 <= n <= 60:
        icos = generate_mackay_icosahedron(L)
        if n < 55:
             # Same removal logic
             dists = np.linalg.norm(icos, axis=1)
             indices_asc = np.argsort(dists)
             selected_indices = indices_asc[:n]
             cand = icos[selected_indices]
             # Add noise
             noise = rng.normal(0, 0.05, cand.shape)
             candidates.append(cand + noise)
        elif n > 55:
             # Add atoms? Just use random start for these.
             pass

    best_energy = np.inf
    best_positions = None
    
    for cand in candidates:
        if len(cand) != n:
            continue
        try:
            energy, positions = relax(cand, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-5, 5, (n, 3))
        
    return best_positions
