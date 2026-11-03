import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate N points in hex lattice
    points = []
    for r in range(-10, 11):
        for q in range(-10, 11):
            dist = np.sqrt(q**2 + q*r + r**2)
            points.append((dist, q, r))
    points.sort()
    
    selected = points[:N]
    
    centers = []
    for _, q, r in selected:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append([x, y])
        
    centers = np.array(centers)
    
    best_S = float('inf')
    best_angle = 0.0
    best_centers = None
    
    # Try rotations of the outer hexagon (and inner)
    # Step 5 degrees for precision, or 1 degree? 1 degree is fast enough for 11 hexagons.
    for alpha_deg in range(0, 60, 1):
        alpha = np.radians(alpha_deg)
        
        # Generate vertices for all inner hexagons with angle alpha
        all_vertices = []
        for cx, cy in centers:
            for k in range(6):
                theta = alpha + np.radians(60 * k)
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                all_vertices.append([vx, vy])
                
        all_vertices = np.array(all_vertices)
        
        # Center the configuration
        # Use centroid of vertices as the center of the outer hexagon
        cx_center = np.mean(all_vertices[:, 0])
        cy_center = np.mean(all_vertices[:, 1])
        
        all_vertices[:, 0] -= cx_center
        all_vertices[:, 1] -= cy_center
        
        # Compute max projection against outer hexagon normals
        # Outer hexagon has vertices at alpha, alpha+60...
        # Normals are at alpha + 30, alpha + 90...
        max_proj = 0
        for k in range(6):
            theta = alpha + np.radians(30 + 60 * k)
            nx = np.cos(theta)
            ny = np.sin(theta)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # r_out = max_proj
        # S = 2 * r_out / sqrt(3)
        S = 2.0 * max_proj / np.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_angle = alpha_deg
            # Store the centered vertices? No, we need centers.
            # The centers were shifted by (cx_center, cy_center) relative to the original lattice coords?
            # No, all_vertices were generated from original centers, then shifted.
            # So the "shifted centers" are original centers - (cx_center, cy_center).
            best_centers = centers - np.array([cx_center, cy_center])

    # Format output
    inner = []
    for cx, cy in best_centers:
        inner.append([float(cx), float(cy), float(best_angle)])
        
    return inner, (0.0, 0.0), float(best_S), float(best_angle)
# EVOLVE-BLOCK-END
