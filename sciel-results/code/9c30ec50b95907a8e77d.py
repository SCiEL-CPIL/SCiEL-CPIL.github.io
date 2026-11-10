import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a constructive approach with analytical calculation of the minimal enclosing hexagon for specific lattice clusters.
    """
    n = 12
    
    # Generate candidate clusters of 12 hexagons in a hexagonal grid
    clusters = []
    
    # Helper to convert axial coords to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        return (x, y)
        
    # Cluster 1: 4x3 Grid
    # r in [-1, 0, 1], q in [-2, -1, 0, 1]
    centers_4x3 = []
    for r in [-1, 0, 1]:
        for q in [-2, -1, 0, 1]:
            centers_4x3.append(axial_to_cartesian(q, r))
    clusters.append(centers_4x3)
    
    # Cluster 2: 3x4 Grid
    # r in [-2, -1, 0, 1], q in [-1, 0, 1]
    centers_3x4 = []
    for r in [-2, -1, 0, 1]:
        for q in [-1, 0, 1]:
            centers_3x4.append(axial_to_cartesian(q, r))
    clusters.append(centers_3x4)
    
    # Cluster 3: Compact subset of Radius 2 Hexagon (19 items)
    # Generate all points in radius 2 hexagon
    full_hex = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                full_hex.append(axial_to_cartesian(q, r))
    # Sort by distance from center and take the 12 closest
    full_hex_sorted = sorted(full_hex, key=lambda p: p[0]**2 + p[1]**2)
    clusters.append(full_hex_sorted[:12])
    
    # Cluster 4: 2x6 Strip (just in case)
    centers_2x6 = []
    for r in [0, 1]:
        for q in range(6):
            # Center it roughly
            q_centered = q - 2.5
            centers_2x6.append(axial_to_cartesian(q_centered, r))
    clusters.append(centers_2x6)

    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    for centers in clusters:
        if len(centers) != 12:
            continue
            
        # Get all vertices of all inner hexagons
        all_vertices = []
        for cx, cy in centers:
            # Vertices of unit hexagon at cx, cy, angle 0
            angles = np.deg2rad(np.arange(6) * 60)
            vx = cx + np.cos(angles)
            vy = cy + np.sin(angles)
            for i in range(6):
                all_vertices.append((vx[i], vy[i]))
        all_vertices = np.array(all_vertices)
        
        min_s_for_cluster = float('inf')
        best_theta_for_cluster = 0.0
        
        # Optimize outer angle (0 to 60 degrees)
        # Sample finely
        for theta_deg in np.linspace(0, 60, 121):
            theta = np.deg2rad(theta_deg)
            # The outward normals of the outer hexagon are at angles theta + 30 + 60*i
            normal_angles = theta + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            
            # Project all inner vertices onto these normals
            # proj[i, j] is projection of vertex i onto normal j
            proj = all_vertices @ normals.T
            # For each normal direction, find the maximum projection (furthest vertex)
            max_projs = np.max(proj, axis=0)
            
            # The condition for containment is:
            # max_proj <= s * apothem_outer
            # apothem_outer = s * sqrt(3) / 2
            # So s >= max_proj / (sqrt(3) / 2)
            required_s = np.max(max_projs) / (np.sqrt(3) / 2)
            
            if required_s < min_s_for_cluster:
                min_s_for_cluster = required_s
                best_theta_for_cluster = theta_deg
                
        if min_s_for_cluster < best_s:
            best_s = min_s_for_cluster
            best_inner = np.array([(cx, cy, 0.0) for cx, cy in centers])
            best_outer_angle = best_theta_for_cluster

    # If no cluster found (should not happen), fallback
    if best_inner is None:
        # Fallback to a simple valid packing
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        best_outer_angle = 0.0
        best_s = 8.0

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, outer_hex_data, best_s
