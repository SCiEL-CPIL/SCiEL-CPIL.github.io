import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    points = []
    # Search in a bounding box large enough to contain 11 points
    # 11 points fit in a hexagon of radius 2 (axial)
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Calculate distance squared (squared Euclidean)
            # x^2 + y^2 = 3(q + r/2)^2 + 9/4 r^2 = 3(q^2 + qr + r^2/4) + 9/4 r^2
            # = 3q^2 + 3qr + 3/4 r^2 + 9/4 r^2 = 3q^2 + 3qr + 3r^2
            dist_sq = 3 * (q*q + q*r + r*r)
            points.append((dist_sq, q, r))
            
    # Sort by distance
    points.sort(key=lambda x: x[0])
    
    # Take the first N points
    selected = points[:N]
    
    # Convert to Cartesian
    inner = []
    max_proj = 0.0
    for dist_sq, q, r in selected:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
        # Calculate projections for constraint check
        # Normals at 90, 30, -30 degrees (for flat-top hexagon? No, pointy top?)
        # Let's assume the outer hexagon is "flat-top" (sides horizontal).
        # Normals are at 90, 30, -30, 270, 210, 150.
        # Unique directions: 90, 30, -30.
        
        # Projection on Y (90 deg): |y|
        p0 = abs(y)
        
        # Projection on 30 deg: x*cos(30) + y*sin(30)
        p1 = abs(x * np.cos(np.pi/6) + y * np.sin(np.pi/6))
        
        # Projection on -30 deg: x*cos(-30) + y*sin(-30)
        p2 = abs(x * np.cos(-np.pi/6) + y * np.sin(-np.pi/6))
        
        max_proj = max(max_proj, p0, p1, p2)
        
    # max_proj is the required R (distance from center to side of outer hex)
    # R = (sqrt(3)/2) * (L - 1)
    # L - 1 = 2 * R / sqrt(3)
    side = 1.0 + 2.0 * max_proj / np.sqrt(3)
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
