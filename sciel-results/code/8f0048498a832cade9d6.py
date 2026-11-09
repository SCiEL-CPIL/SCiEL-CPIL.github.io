<<<<<<< SEARCH
    # Try different outer hexagon rotations to minimize side length
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    
    # For a given outer hexagon rotation, the minimal side length to contain a set of points
    # is determined by the maximum "distance" in the 6 normal directions
    # For a regular hexagon with center (cx,cy), side s, rotation a:
    # The 6 face normals are at angles a + i*pi/3
    # A point (px,py) is inside if for all i: 
    #   (px-cx)*cos(n_i) + (py-cy)*sin(n_i) <= s * sqrt(3)/2
    # So s >= max over all points and all i of [(px-cx)*cos(n_i) + (py-cy)*sin(n_i)] / (sqrt(3)/2)
    
    for outer_angle in np.linspace(0, 60, 121):  # 0 to 60 degrees
        a = np.radians(outer_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_verts[:, 0] - cx_mean
        dy = all_verts[:, 1] - cy_mean
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
=======
    # Try different outer hexagon rotations to minimize side length
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    
    # For a given outer hexagon rotation, the minimal side length to contain a set of points
    # is determined by the maximum "distance" in the 6 normal directions
    # For a regular hexagon with center (cx,cy), side s, rotation a:
    # The 6 face normals are at angles a + i*pi/3
    # A point (px,py) is inside if for all i: 
    #   (px-cx)*cos(n_i) + (py-cy)*sin(n_i) <= s * sqrt(3)/2
    # So s >= max over all points and all i of [(px-cx)*cos(n_i) + (py-cy)*sin(n_i)] / (sqrt(3)/2)
    
    # Coarse search over angles
    for outer_angle in np.linspace(0, 60, 121):  # 0 to 60 degrees
        a = np.radians(outer_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_verts[:, 0] - cx_mean
        dy = all_verts[:, 1] - cy_mean
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
    
    # Fine search around best angle
    if best_side < float('inf'):
        for outer_angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 201):
            a = np.radians(outer_angle)
            normals = []
            for i in range(6):
                theta = a + np.pi/6 + i * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            dx = all_verts[:, 0] - cx_mean
            dy = all_verts[:, 1] - cy_mean
            
            projections = np.zeros((N * 6, 6))
            for i in range(6):
                projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
            
            max_proj = projections.max(axis=0)
            min_proj = projections.min(axis=0)
            required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
            side = required_s.max()
            
            if side < best_side:
                best_side = side
                best_center = (cx_mean, cy_mean)
                best_angle = outer_angle
    
    # Optimize center for the best angle
    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    # The optimal center minimizes max over i of (max_proj_i - min_proj_i) / 2
    # where proj_i = dot(point - center, normal_i)
    # This is equivalent to: for each normal direction, the width is (max - min) of projections
    # The side length is max_i width_i / sqrt(3)
    # To minimize this, we want to center the point set optimally
    # For a given angle, the optimal center is the midpoint of the projection extremes in the "tightest" direction
    # Let's just try a few candidate centers
    
    # Candidate 1: mean of vertices
    # Candidate 2: center of bounding box in each normal direction
    for trial_cx, trial_cy in [(cx_mean, cy_mean),
                                 (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2,
                                 (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2,
                                 (all_verts[:, 0].min() + all_verts[:, 0].max()) / 2,
                                 (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2]:
        dx = all_verts[:, 0] - trial_cx
        dy = all_verts[:, 1] - trial_cy
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (trial_cx, trial_cy)
>>>>>>> REPLACE

