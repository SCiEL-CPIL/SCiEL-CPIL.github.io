import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for hexagon centers
    # Distance between centers of touching hexagons is sqrt(3)
    d = np.sqrt(3)
    e1 = np.array([d, 0.0])
    e2 = np.array([d/2.0, 1.5])
    
    # Define candidate topologies as lists of lattice coordinates (i, j)
    # Position = i * e1 + j * e2
    
    topologies = []
    
    # 1. Linear chain
    line_coords = [(k, 0) for k in range(-5, 6)]
    topologies.append(line_coords)
    
    # 2. Compact cluster (Center + 6 neighbors + 4 more)
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Variation 1
    comp1 = base + [(2,0), (1,1), (-1,2), (-2,1)]
    topologies.append(comp1)
    
    # Variation 2
    comp2 = base + [(2,0), (2,-1), (-2,0), (-2,1)]
    topologies.append(comp2)
    
    # Variation 3: 3x3 block minus corners + 2
    comp3 = [(i,j) for i in range(-1,2) for j in range(-1,2) if not (i==1 and j==1 or i==-1 and j==-1)] + [(2,0), (-2,0)]
    topologies.append(comp3)
    
    # Variation 4: "Barbell" or elongated
    comp4 = [(k,0) for k in range(-2,3)] + [(1,1), (-1,1), (1,-1), (-1,-1)]
    topologies.append(comp4)

    best_S = float('inf')
    best_result = None
    
    for coords in topologies:
        # Generate centers
        centers = np.array([i * e1 + j * e2 for i, j in coords])
        
        # Try two inner orientations: aligned with lattice (0) and rotated 30 deg
        for inner_angle in [0.0, np.pi/6]:
            # Vertices of a unit hexagon with angle 0
            base_angles = np.arange(6) * np.pi / 3.0
            base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
            
            # Rotate base vertices by inner_angle
            c_ang = np.cos(inner_angle)
            s_ang = np.sin(inner_angle)
            rot_mat = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
            verts = base_verts @ rot_mat.T
            
            # All points: centers + verts
            # centers: (11, 1, 2)
            # verts: (1, 6, 2)
            # sum: (11, 6, 2)
            all_points = np.sum(centers[:, np.newaxis, :] + verts[np.newaxis, :, :], axis=0)
            all_points = all_points.reshape(-1, 2)
            
            # Find minimal enclosing regular hexagon for all_points
            min_S_for_topo = float('inf')
            best_theta = 0.0
            best_center = np.array([0.0, 0.0])
            
            # Scan theta
            thetas = np.linspace(0, np.pi/3, 180, endpoint=False)
            for theta in thetas:
                normals = []
                for k in range(6):
                    ang = theta + k * np.pi / 3.0
                    normals.append(np.array([np.cos(ang), np.sin(ang)]))
                normals = np.array(normals) # (6, 2)
                
                # Project points onto normals
                projections = normals @ all_points.T # (6, 66)
                
                mins = np.min(projections, axis=1)
                maxs = np.max(projections, axis=1)
                widths = maxs - mins
                
                S_candidate = np.max(widths) / np.sqrt(3)
                
                if S_candidate < min_S_for_topo:
                    min_S_for_topo = S_candidate
                    best_theta = theta
                    
                    # Calculate optimal center
                    mids = (mins + maxs) / 2.0
                    A = normals
                    ATA = A.T @ A
                    ATm = A.T @ mids
                    try:
                        c_candidate = np.linalg.solve(ATA, ATm)
                    except np.linalg.LinAlgError:
                        c_candidate = np.linalg.lstsq(A, mids, rcond=None)[0]
                    best_center = c_candidate
            
            if min_S_for_topo < best_S:
                best_S = min_S_for_topo
                angle_deg = np.degrees(inner_angle)
                inner_list = []
                for c in centers:
                    inner_list.append([c[0], c[1], angle_deg])
                
                best_result = (inner_list, (best_center[0], best_center[1]), best_S, np.degrees(best_theta))
    
    if best_result:
        return best_result
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
