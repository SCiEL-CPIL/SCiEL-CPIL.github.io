import numpy as np

N = 12

def construct():
    # Hexagonal grid positions for 12 unit hexagons
    # Using axial coordinates (q, r)
    hex_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1)
    ]
    
    # Convert to Cartesian coordinates
    centers = []
    for q, r in hex_positions:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # All inner hexagons have angle 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for x, y, _ in inner:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = x + np.cos(angle)
            vy = y + np.sin(angle)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # Optimize over center and rotation
    # Start with centroid as initial center
    centroid = np.mean(all_vertices, axis=0)
    
    def required_side_length(cx, cy, theta, vertices):
        max_proj = 0.0
        for k in range(6):
            angle = theta + np.radians(60 * k)
            dx = vertices[:, 0] - cx
            dy = vertices[:, 1] - cy
            proj = dx * np.cos(angle) + dy * np.sin(angle)
            max_proj = max(max_proj, np.max(proj))
        return 2.0 * max_proj / np.sqrt(3)
    
    best_s = float('inf')
    best_theta = 0.0
    best_cx, best_cy = centroid
    
    # Try different centers and rotations
    # Use a grid search around the centroid
    for dx in np.linspace(-1, 1, 21):
        for dy in np.linspace(-1, 1, 21):
            cx = centroid[0] + dx
            cy = centroid[1] + dy
            for theta_deg in range(0, 60, 2):
                theta = np.radians(theta_deg)
                s = required_side_length(cx, cy, theta, all_vertices)
                if s < best_s:
                    best_s = s
                    best_theta = theta
                    best_cx, best_cy = cx, cy
    
    return inner, (best_cx, best_cy), best_s, np.degrees(best_theta)
