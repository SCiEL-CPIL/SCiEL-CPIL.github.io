import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Unit hexagon vertices (flat-top orientation)
    # A unit regular hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # with circumradius = 1
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # For a cluster of unit hexagons on a hexagonal lattice,
    # the center-to-center distance for adjacent hexagons is sqrt(3)
    # (since the inradius of a unit hexagon is sqrt(3)/2, and adjacent hexagons
    #  share an edge, so centers are 2*inradius = sqrt(3) apart)
    
    # Hexagonal lattice basis vectors
    # In a hexagonal lattice with spacing d = sqrt(3):
    # e1 = (d, 0) = (sqrt(3), 0)
    # e2 = (d/2, d*sqrt(3)/2) = (sqrt(3)/2, 3/2)
    d = np.sqrt(3)
    e1 = np.array([d, 0.0])
    e2 = np.array([d/2, 3.0/2])
    
    # Generate hexagonal cluster positions
    # For N=11, use a compact hexagonal arrangement
    # Hexagonal numbers: 1, 7, 19, ...
    # 11 = 1 + 6 + 4, so we need a non-perfect hexagon
    # Let's try different cluster shapes and pick the best
    
    def get_cluster_positions(shape):
        """shape is a list of (i, j) in hex lattice coordinates"""
        positions = []
        for (i, j) in shape:
            pos = i * e1 + j * e2
            positions.append(pos)
        return np.array(positions)
    
    # Try several cluster arrangements for 11 hexagons
    # Shape 1: 3-4-3-1 or similar
    # Let's enumerate reasonable compact shapes
    
    # Hex lattice coordinates: position = i*e1 + j*e2
    # Neighbors of (0,0): (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    
    # Compact shapes for 11:
    # Option A: Center + ring of 6 + 4 more
    # The ring of 6 around center: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    # Add 4 more: extend in some directions
    
    shapes = []
    
    # Shape 1: 1 + 6 + 4 (extend the hexagon in one direction)
    # Center (0,0), ring: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    # Add: (2,0), (2,-1), (1,1), (-1,1) -- wait, (-1,1) is already in ring
    # Let me be more careful
    
    # Ring of 6 around (0,0):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # These are the 6 neighbors in axial coordinates
    
    # For 11, let's try:
    # 3 rows: 3-5-3 or 4-4-3 or 3-4-4 etc.
    
    # Let me just try to minimize the enclosing hexagon by searching over cluster shapes
    
    # Actually, for a given cluster, the minimal enclosing regular hexagon
    # (aligned with the lattice) has side length = max over 6 directions of
    # (max projection of all vertices in that direction) + 1
    # where the projection is along the direction perpendicular to each face
    
    # The 6 face-normal directions of a regular hexagon (flat-top) are at
    # 30, 90, 150, 210, 270, 330 degrees
    # But for our lattice-aligned hexagon, the normals are at 0, 60, 120, 180, 240, 300
    # Wait, need to be careful about orientation.
    
    # Let me define: a regular hexagon with "pointy-top" has vertices at 30,90,...
    # A "flat-top" has vertices at 0,60,120,...
    # The face normals for flat-top are at 30,90,150,210,270,330
    
    # For containment: a point (x,y) is inside a flat-top regular hexagon of side S
    # centered at origin iff:
    # |x| <= S and |x/2 + y*sqrt(3)/2| <= S*sqrt(3)/2 ... 
    # Actually, let me use the standard formulation.
    
    # A regular hexagon with side S, flat-top, centered at origin:
    # Vertices at (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2,-S*sqrt(3)/2), (S/2,-S*sqrt(3)/2)
    # The 6 faces have outward normals at angles 30, 90, 150, 210, 270, 330 degrees
    # The constraint is: for each face normal n_k, dot(point, n_k) <= S * cos(30) = S*sqrt(3)/2
    # Wait, the apothem is S*sqrt(3)/2, so the constraint is dot(point, n_k) <= S*sqrt(3)/2
    
    # So for a cluster with vertices V, the minimal S (flat-top, aligned) is:
    # S = max over all vertices v and all 6 normals n_k of dot(v, n_k) / (sqrt(3)/2)
    #   = (2/sqrt(3)) * max_{v, k} dot(v, n_k)
    
    # But we can also rotate the outer hexagon! The task allows rotation.
    # So we need to minimize over the outer hexagon's rotation angle theta.
    
    # For a given rotation theta, the 6 face normals are at theta + 30 + k*60 degrees
    # S(theta) = (2/sqrt(3)) * max_{v, k} dot(v, n_k(theta))
    # We want min_theta S(theta)
    
    # This is a 1D optimization problem!
    
    def compute_min_side(cluster_centers):
        """Given cluster center positions, compute minimal enclosing hexagon side length."""
        # Get all vertices of all hexagons
        all_vertices = []
        for center in cluster_centers:
            verts = center + hex_vertices  # unit hexagon vertices
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # The 6 face normal directions for a flat-top hexagon are at 30+60k degrees
        # For rotation theta, normals are at theta + 30 + 60k degrees
        base_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        
        def side_length(theta):
            normal_angles = theta + base_normal_angles
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            # For each vertex and each normal, compute dot product
            # dot(v, n) for all v in all_vertices, all n in normals
            # = all_vertices @ normals.T
            dots = all_vertices @ normals.T
            max_dot = np.max(dots)
            return max_dot / (np.sqrt(3) / 2)
        
        # Golden section search over theta in [0, 60 degrees] (period is 60 degrees)
        # Actually, the period is 60 degrees due to hexagonal symmetry
        a, b = 0.0, np.pi / 3.0
        gr = (np.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = side_length(c)
        fd = side_length(d)
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = side_length(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = side_length(d)
        
        theta_opt = (a + b) / 2
        return side_length(theta_opt), theta_opt
    
    # Define cluster shapes
    # Shape 1: compact 11-hexagon cluster
    # Let's try a few and pick the best
    
    # Shape A: 3-4-3-1 (rows in hex lattice)
    # In axial coords (i,j), row j has i ranging...
    # Let me just define explicit sets of (i,j) coordinates
    
    # The 6 neighbors of (0,0) in axial coords: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
    
    # Shape 1: center + 6 neighbors + 4 more
    # Add (2,0), (2,-1), (1,1), (-2,1) -- let me check if these are valid extensions
    shape1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
               (2,0),(2,-1),(1,1),(-2,1)]
    
    # Shape 2: different extension
    shape2 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
               (2,0),(2,-1),(1,1),(0,2)]
    
    # Shape 3: more symmetric
    shape3 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
               (2,0),(-2,0),(0,2),(0,-2)]
    
    # Shape 4: 4-3-4 arrangement
    shape4 = [(-1,0),(0,0),(1,0),(2,0),
              (-1,1),(0,1),(1,1),
              (-1,-1),(0,-1),(1,-1),(2,-1)]
    
    # Shape 5: 3-5-3
    shape5 = [(0,0),(1,0),(2,0),
              (-1,1),(0,1),(1,1),(2,1),(3,1),
              (0,-1),(1,-1),(2,-1)]
    
    all_shapes = [shape1, shape2, shape3, shape4, shape5]
    
    best_side = float('inf')
    best_theta = 0.0
    best_shape = None
    
    for shape in all_shapes:
        centers = get_cluster_positions(shape)
        side, theta = compute_min_side(centers)
        if side < best_side:
            best_side = side
            best_theta = theta
            best_shape = shape
    
    # Also try translating the cluster to center it optimally
    # The optimal center is the centroid of the cluster's bounding box
    # But since we're minimizing over rotation, let's also optimize the center
    
    # Actually, for a given cluster and rotation, the optimal center is the
    # center of the minimal enclosing hexagon, which we can compute.
    # But the side length formula above assumes center at origin.
    # We need to also optimize the translation.
    
    # For a given rotation theta, the optimal center c minimizes:
    # max_{v, k} dot(v - c, n_k) / (sqrt(3)/2)
    # This is equivalent to finding c that minimizes the maximum of
    # (max_v dot(v,n_k) - dot(c,n_k)) and (max_v dot(-v,n_k) + dot(c,n_k))
    # for each k.
    
    # Actually, the minimal enclosing hexagon centered at c has side:
    # S = max_k [ (max_v dot(v,n_k) - min_v dot(v,n_k)) / 2 ] / (sqrt(3)/2)
    # = (1/sqrt(3)) * max_k [max_v dot(v,n_k) - min_v dot(v,n_k)]
    
    # This is independent of c! The center c is just the midpoint of the range.
    # So the side length only depends on the width of the cluster in each normal direction.
    
    def compute_min_side_v2(cluster_centers):
        """Compute minimal enclosing hexagon side length, optimizing over rotation."""
        all_vertices = []
        for center in cluster_centers:
            verts = center + hex_vertices
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        base_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        
        def side_length(theta):
            normal_angles = theta + base_normal_angles
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            dots = all_vertices @ normals.T
            # For each normal direction, compute the width
            widths = np.max(dots, axis=0) - np.min(dots, axis=0)
            max_width = np.max(widths)
            return max_width / np.sqrt(3)
        
        # Golden section search
        a, b = 0.0, np.pi / 3.0
        gr = (np.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = side_length(c)
        fd = side_length(d)
        
        for _ in range(60):
            if time.time() > deadline - 10:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = side_length(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = side_length(d)
        
        theta_opt = (a + b) / 2
        return side_length(theta_opt), theta_opt
    
    best_side = float('inf')
    best_theta = 0.0
    best_centers = None
    
    for shape in all_shapes:
        centers = get_cluster_positions(shape)
        side, theta = compute_min_side_v2(centers)
        if side < best_side:
            best_side = side
            best_theta = theta
            best_centers = centers
    
    # Compute the optimal center for the best configuration
    # The center is the midpoint of the cluster's extent in each direction
    all_vertices = []
    for center in best_centers:
        verts = center + hex_vertices
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    base_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normal_angles = best_theta + base_normal_angles
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    dots = all_vertices @ normals.T
    max_dots = np.max(dots, axis=0)
    min_dots = np.min(dots, axis=0)
    
    # The center in each normal direction is (max+min)/2
    # We need to find c such that dot(c, n_k) = (max_dots[k] + min_dots[k])/2 for all k
    # This is a system of 6 equations in 2 unknowns, so it's overdetermined.
    # Use least squares.
    targets = (max_dots + min_dots) / 2
    # normals.T @ c = targets
    c_opt, _, _, _ = np.linalg.lstsq(normals.T, targets, rcond=None)
    
    # Build the inner hexagon list
    inner = []
    for center in best_centers:
        inner.append([float(center[0]), float(center[1]), 0.0])
    
    outer_center = (float(c_opt[0]), float(c_opt[1]))
    outer_side = float(best_side)
    outer_rotation = float(np.degrees(best_theta))
    
    return inner, outer_center, outer_side, outer_rotation

