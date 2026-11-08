import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define hexagonal lattice vectors
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def get_min_side_length(axial_coords, outer_rotation_deg=0.0):
        # Convert to Cartesian centers
        centers = [axial_to_cartesian(q, r) for q, r in axial_coords]
        
        # Generate vertices for each unit hexagon
        # Unit hexagon vertices (angle 0 aligned with x-axis)
        hex_verts = []
        for k in range(6):
            angle = np.deg2rad(60 * k)
            hex_verts.append((np.cos(angle), np.sin(angle)))
        hex_verts = np.array(hex_verts)
        
        all_vertices = []
        for cx, cy in centers:
            all_vertices.append(cx + hex_verts[:, 0])
            all_vertices.append(cy + hex_verts[:, 1])
        
        all_vertices = np.array(list(zip(all_vertices[0], all_vertices[1])))
        
        # Minimize side length over outer rotation
        # The outer hexagon normals are at phi + k*60
        # We check phi in [0, 30]
        min_s = float('inf')
        # Sample rotations
        for phi_deg in np.linspace(0, 30, 31):
            phi = np.deg2rad(phi_deg)
            max_width_ratio = 0
            for k in range(6):
                angle = phi + np.deg2rad(60 * k)
                nx = np.cos(angle)
                ny = np.sin(angle)
                projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
                width = np.max(projs) - np.min(projs)
                # S * sqrt(3)/2 = width / 2 => S = width / sqrt(3)
                s = width / np.sqrt(3)
                if s > max_width_ratio:
                    max_width_ratio = s
            if max_width_ratio < min_s:
                min_s = max_width_ratio
        
        return min_s

    # Candidate 1: 3x4 Rhombus
    # i in [-1, 0, 1], j in [-1.5, -0.5, 0.5, 1.5]
    # Axial coords are integers.
    # Let's use integer axial coords.
    # j values: -2, -1, 0, 1? No, 4 rows.
    # Let's use j in [0, 1, 2, 3] and i in [0, 1, 2].
    # Center it.
    rhombus_34 = []
    for i in range(3):
        for j in range(4):
            rhombus_34.append((i, j))
    
    # Candidate 2: Flower (7) + 5
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-3)?
    # Let's pick 5 that make it compact.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -> This is a "C" shape or arc.
    flower_12 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (1,1), (0,2), (-1,2), (-2,1)]
                 
    # Candidate 3: Another variation
    # Fill the other side?
    flower_12_v2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,0), (1,1), (0,2), (-1,2), (-2,0)] # (-2,0) instead of (-2,1)?
                 
    # Evaluate
    best_score = float('inf')
    best_config = rhombus_34
    
    configs = [rhombus_34, flower_12, flower_12_v2]
    
    for config in configs:
        s = get_min_side_length(config)
        if s < best_score:
            best_score = s
            best_config = config
            
    # Convert best_config to Cartesian for output
    # The function requires inner rows (x, y, angle_degrees)
    # We used angle 0 for inner hexagons.
    inner = []
    for q, r in best_config:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # Outer center is (0,0) because we centered the lattice?
    # No, the lattice was not centered in Cartesian.
    # We should center the configuration.
    # Calculate centroid of centers
    centers = [axial_to_cartesian(q, r) for q, r in best_config]
    cx_avg = np.mean([c[0] for c in centers])
    cy_avg = np.mean([c[1] for c in centers])
    
    # Shift all points
    for i in range(len(inner)):
        inner[i][0] -= cx_avg
        inner[i][1] -= cy_avg
        
    # Outer center is (0,0)
    # Outer side length is best_score
    # Outer rotation: We need to find the optimal rotation used in get_min_side_length
    # Let's re-calculate the optimal rotation for the best config
    # ... (simpler to just return 0.0 if we assume symmetry, or re-run the loop)
    
    # For now, return 0.0 rotation. The evaluator might handle rotation?
    # No, we must return the rotation.
    # Let's find the optimal rotation.
    
    def get_optimal_rotation(axial_coords):
        centers = [axial_to_cartesian(q, r) for q, r in axial_coords]
        hex_verts = np.array([(np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))) for k in range(6)])
        all_vertices = []
        for cx, cy in centers:
            all_vertices.append(cx + hex_verts[:, 0])
            all_vertices.append(cy + hex_verts[:, 1])
        all_vertices = np.array(list(zip(all_vertices[0], all_vertices[1])))
        
        best_rot = 0.0
        best_s = float('inf')
        for phi_deg in np.linspace(0, 30, 31):
            phi = np.deg2rad(phi_deg)
            max_w = 0
            for k in range(6):
                angle = phi + np.deg2rad(60 * k)
                nx = np.cos(angle)
                ny = np.sin(angle)
                projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
                width = np.max(projs) - np.min(projs)
                s = width / np.sqrt(3)
                if s > max_w: max_w = s
            if max_w < best_s:
                best_s = max_w
                best_rot = phi_deg
        return best_rot, best_s

    rot, side = get_optimal_rotation(best_config)
    
    return inner, (0.0, 0.0), side, rot

# EVOLVE-BLOCK-END
