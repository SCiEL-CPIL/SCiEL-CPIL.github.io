import numpy as np

def construct():
    # N = 11
    # Define centers
    centers = [(0.0, 0.0)]
    # 6 neighbors
    for i in range(6):
        angle = np.deg2rad(i * 60)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
    # 4 corners (from the next layer)
    # The next layer corners are at distance 2*sqrt(3)
    # Angles 0, 60, 120, 180
    for i in [0, 1, 2, 3]:
        angle = np.deg2rad(i * 60)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        
    # Inner hexagons are rotated by -30 degrees (so sides face 0, 60, ...)
    # This minimizes the "vertex" radius in the 0, 60 directions
    inner_angle_deg = -30.0
    
    # Generate all vertices of all inner hexagons
    points = []
    for cx, cy in centers:
        for j in range(6):
            # Vertex angle relative to center
            v_angle = np.deg2rad(inner_angle_deg + j * 60)
            vx = cx + np.cos(v_angle)
            vy = cy + np.sin(v_angle)
            points.append((vx, vy))
            
    points = np.array(points)
    
    # Find minimal enclosing regular hexagon
    # Search container rotation phi in [0, 60)
    best_S = float('inf')
    best_phi = 0.0
    
    # Sample phi
    phis = np.linspace(0, 60, 1000)
    for phi_deg in phis:
        phi = np.deg2rad(phi_deg)
        # Calculate required S for this rotation
        # f(alpha) = (sqrt(3)/2) / cos((alpha mod 60) - 30)
        # alpha is angle of point relative to container axes
        alphas = np.arctan2(points[:, 1], points[:, 0]) - phi
        # Normalize alphas to [-30, 30] relative to side normals?
        # Container vertices are at phi + 0, 60, ...
        # Container sides are at phi + 30, 90, ...
        # Distance to side at angle alpha (relative to vertex 0)
        # The boundary is defined by lines x cos(30) + y sin(30) <= A?
        # Let's use the formula: r <= S * (sqrt(3)/2) / cos(delta)
        # where delta is the angle from the nearest side normal.
        # Side normals are at phi + 30 + k*60.
        # Angle from point to nearest side normal:
        delta = np.mod(alphas - np.deg2rad(30), np.deg2rad(60)) - np.deg2rad(30)
        # delta is in [-30, 30]
        cos_delta = np.cos(delta)
        # Avoid division by zero or negative cos (shouldn't happen if within 30 deg)
        # f(alpha) = (sqrt(3)/2) / cos(delta)
        # r_point / f(alpha) <= S
        # S >= r_point * cos(delta) / (sqrt(3)/2)
        r_points = np.linalg.norm(points, axis=1)
        S_required = r_points * cos_delta / (np.sqrt(3) / 2)
        S_max = np.max(S_required)
        if S_max < best_S:
            best_S = S_max
            best_phi = phi_deg
            
    # Return the inner hexagons, center, side, angle
    # inner: list of (x, y, angle_degrees)
    inner = [[cx, cy, inner_angle_deg] for cx, cy in centers]
    center = (0.0, 0.0)
    side = best_S
    angle = best_phi
    
    return inner, center, side, angle
