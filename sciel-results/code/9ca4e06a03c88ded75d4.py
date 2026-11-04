import numpy as np

def construct():
    N = 12
    
    dx = np.sqrt(3)
    dy = 1.5
    
    # Generate base 3x4 grid centers
    base_centers = []
    for r in range(3):
        for c in range(4):
            x = c * dx + (r % 2) * (dx / 2)
            y = r * dy
            base_centers.append([x, y])
            
    # Center the base grid
    xs = [c[0] for c in base_centers]
    ys = [c[1] for c in base_centers]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    centered_base = np.array([[c[0] - cx, c[1] - cy] for c in base_centers])
    
    def get_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + np.cos(theta)
            y = cy + np.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def min_container_side(centers, inner_angle):
        # Container angle 0
        normals = []
        for i in range(6):
            theta = np.deg2rad(30 + i * 60)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        max_proj = 0
        for c in centers:
            verts = get_vertices(c[0], c[1], inner_angle)
            # Project all verts onto all normals
            # verts shape (6,2), normals shape (6,2)
            # We want max |v . n|
            # Use broadcasting
            # (6,1) dot (1,2) -> (6,2) ? No.
            # verts @ normals.T -> (6,6) matrix of projections
            projs = np.abs(verts @ normals.T)
            max_proj = max(max_proj, np.max(projs))
            
        S = max_proj / (np.sqrt(3) / 2)
        return S

    best_S = float('inf')
    best_grid_rot = 0
    best_inner_angle = 0
    
    # Search grid rotations and inner angles
    for grid_rot_deg in range(0, 180, 15):
        grid_rot_rad = np.deg2rad(grid_rot_deg)
        cos_r = np.cos(grid_rot_rad)
        sin_r = np.sin(grid_rot_rad)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rotated_centers = centered_base @ rot_matrix.T
        
        for inner_angle in [0, 30, 60, 90, 120, 150]:
            S = min_container_side(rotated_centers, inner_angle)
            if S < best_S:
                best_S = S
                best_grid_rot = grid_rot_deg
                best_inner_angle = inner_angle
                
    # Reconstruct best centers
    grid_rot_rad = np.deg2rad(best_grid_rot)
    cos_r = np.cos(grid_rot_rad)
    sin_r = np.sin(grid_rot_rad)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    final_centers = centered_base @ rot_matrix.T
    
    inner = []
    for c in final_centers:
        inner.append([float(c[0]), float(c[1]), float(best_inner_angle)])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
