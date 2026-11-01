<<<<<<< SEARCH
def make_mackay_icosahedron(n_atoms):
    """
    Generate a Mackay icosahedron cluster.
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms at distance r1
    Shell 2: 42 atoms at distance r2
    Total: 55 atoms
    """
    # Icosahedron vertices (unit length)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Shell 1: 12 atoms at distance r1
    r1 = 1.0  # Will be optimized, but start with reasonable value
    shell1 = vertices * r1
    
    # Shell 2: 42 atoms
    # The 42 positions of the second shell in a Mackay icosahedron are:
    # 12 face centers (distance ~r2_face)
    # 30 edge midpoints (distance ~r2_edge)
    
    # Actually, for a proper Mackay icosahedron with 55 atoms:
    # Shell 0: 1 atom
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms
    # The 42 atoms in shell 2 are arranged as:
    # - 12 atoms at face centers of the icosahedron
    # - 30 atoms at edge midpoints of the icosahedron
    # Total = 12 + 30 = 42 ✓
    
    # Icosahedron edges: pairs of vertices that are closest neighbors
    # In a regular icosahedron, each vertex connects to 5 others
    # There are 30 edges total
    
    # Let's find edges by distance
    dist_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dist_matrix[i, j] = np.linalg.norm(vertices[i] - vertices[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    
    # Find the minimum distance (edge length)
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dist_matrix[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_midpoints = []
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (12 atoms)
    # Icosahedron has 20 triangular faces
    # We need to find the faces
    # A face is a triple of mutually adjacent vertices
    
    faces = []
    # Build adjacency
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Find all triangles
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    triangle = tuple(sorted([i, j, k]))
                    if triangle not in faces:
                        faces.append(triangle)
    
    # Deduplicate
    faces = list(set(faces))
    
    # There should be 20 faces
    face_centers = []
    for face in faces:
        center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Normalize edge midpoints and face centers to unit length, then scale
    edge_norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    face_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # The radial distances for shell 2 in Mackay icosahedron:
    # For a complete two-shell Mackay icosahedron, the ratio r2/r1 is typically around 1.5-1.6
    # Let's use r2 = 1.5 * r1 as starting point
    r2 = 1.5
    
    shell2_face = (face_centers / face_norms) * r2
    shell2_edge = (edge_midpoints / edge_norms) * r2
    
    # Combine all atoms
    atoms = [np.zeros(3)]  # Shell 0
    atoms.extend(list(shell1))  # Shell 1
    atoms.extend(list(shell2_face))  # Shell 2 face centers
    atoms.extend(list(shell2_edge))  # Shell 2 edge midpoints
    
    positions = np.array(atoms[:n_atoms])
    
    # Center the cluster
    positions -= np.mean(positions, axis=0)
    
    return positions
=======
def make_mackay_icosahedron(n_atoms, r1=1.0, r2=1.55):
    """
    Generate a Mackay icosahedron cluster.
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms at distance r1
    Shell 2: 42 atoms at distance r2
    Total: 55 atoms
    """
    # Icosahedron vertices (unit length)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Shell 1: 12 atoms at distance r1
    shell1 = vertices * r1
    
    # Find edges by distance
    dist_matrix = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dist_matrix[i, j] = np.linalg.norm(vertices[i] - vertices[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    
    min_dist = np.min(dist_matrix[dist_matrix > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dist_matrix[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_midpoints = []
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (20 faces)
    faces = []
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    triangle = tuple(sorted([i, j, k]))
                    if triangle not in faces:
                        faces.append(triangle)
    
    faces = list(set(faces))
    
    face_centers = []
    for face in faces:
        center = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Normalize and scale
    edge_norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    face_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    shell2_face = (face_centers / face_norms) * r2
    shell2_edge = (edge_midpoints / edge_norms) * r2
    
    # Combine all atoms
    atoms = [np.zeros(3)]  # Shell 0
    atoms.extend(list(shell1))  # Shell 1
    atoms.extend(list(shell2_face))  # Shell 2 face centers
    atoms.extend(list(shell2_edge))  # Shell 2 edge midpoints
    
    positions = np.array(atoms[:n_atoms])
    positions -= np.mean(positions, axis=0)
    
    return positions
>>>>>>> REPLACE

