import numpy as np

def construct():
    # 3-5-3 hexagon cluster
    # Centers defined in a local coordinate system
    # Row 2 (middle, 5 hexes) at y=0
    # Row 1 (top, 3 hexes) at y=1.5
    # Row 3 (bottom, 3 hexes) at y=-1.5
    
    centers = []
    
    # Middle row (5 hexes)
    # Spacing sqrt(3)
    # Centered at 0
    x_mid = [-2*np.sqrt(3), -np.sqrt(3), 0, np.sqrt(3), 2*np.sqrt(3)]
    for x in x_mid:
        centers.append((x, 0.0))
        
    # Top row (3 hexes)
    # Shifted by sqrt(3)/2
    x_top = [-np.sqrt(3), 0, np.sqrt(3)]
    for x in x_top:
        centers.append((x, 1.5))
        
    # Bottom row (3 hexes)
    x_bot = [-np.sqrt(3), 0, np.sqrt(3)]
    for x in x_bot:
        centers.append((x, -1.5))
        
    # Calculate minimum side length s
    # We assume the container is aligned with the inner hexagons (0 deg rotation)
    # and centered at (0,0)
    
    # Vertices of a unit hexagon at (cx, cy) with 0 rotation
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    offsets = np.array([(np.cos(a), np.sin(a)) for a in angles])
    
    max_s = 0.0
    
    for cx, cy in centers:
        for ox, oy in offsets:
            px = cx + ox
            py = cy + oy
            
            # Check if point (px, py) is inside hexagon of side s
            # Hexagon vertices at (s, 0), (s/2, s*sqrt(3)/2), ...
            # Inequalities:
            # 1. x <= s
            # 2. x >= -s
            # 3. y <= s*sqrt(3)/2
            # 4. y >= -s*sqrt(3)/2
            # 5. y <= -sqrt(3)*x + s*sqrt(3)  (Upper Right)
            # 6. y >= sqrt(3)*x + s*sqrt(3)    (Upper Left) -- Wait, signs
            # Let's use the standard form:
            # For a hexagon with "pointy" left/right (vertices at 0, 180):
            # The sides are:
            # Right: x = s
            # Left: x = -s
            # Top-Right: y = -sqrt(3)(x - s)  => y = -sqrt(3)x + s*sqrt(3)
            # Bottom-Right: y = sqrt(3)(x - s)  => y = sqrt(3)x - s*sqrt(3)
            # Top-Left: y = sqrt(3)(x + s)     => y = sqrt(3)x + s*sqrt(3)
            # Bottom-Left: y = -sqrt(3)(x + s) => y = -sqrt(3)x - s*sqrt(3)
            
            # Constraints:
            # s >= x
            # s >= -x
            # s*sqrt(3)/2 >= y
            # s*sqrt(3)/2 >= -y
            # s*sqrt(3) >= y + sqrt(3)x  => s >= (y + sqrt(3)x) / sqrt(3)
            # s*sqrt(3) >= -y + sqrt(3)x => s >= (-y + sqrt(3)x) / sqrt(3)
            
            s1 = abs(px)
            s2 = abs(py) / (np.sqrt(3)/2)
            s3 = (py + np.sqrt(3)*px) / np.sqrt(3)
            s4 = (-py + np.sqrt(3)*px) / np.sqrt(3)
            
            # Note: s3 and s4 handle the diagonal sides.
            # Actually, let's just use the max of the projections.
            # The distance from origin to the side in direction theta is s*sqrt(3)/2? No.
            # The inradius is r = s*sqrt(3)/2.
            # The support function h(theta) of a hexagon is r / cos(theta - phi) where phi is angle to normal.
            
            # Simpler: Check the 6 inequalities directly.
            # 1. px <= s
            # 2. -px <= s
            # 3. py <= s*sqrt(3)/2
            # 4. -py <= s*sqrt(3)/2
            # 5. py + sqrt(3)*px <= s*sqrt(3)
            # 6. -py + sqrt(3)*px <= s*sqrt(3)
            
            cand = max(
                px,
                -px,
                py / (np.sqrt(3)/2),
                -py / (np.sqrt(3)/2),
                (py + np.sqrt(3)*px) / np.sqrt(3),
                (-py + np.sqrt(3)*px) / np.sqrt(3)
            )
            
            if cand > max_s:
                max_s = cand

    # Add a small epsilon for safety
    s = max_s + 0.001
    
    # Prepare output
    # inner: list of [x, y, angle]
    # angle is 0 for all
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    # outer: center, side, rotation
    # Center is (0,0)
    # Rotation is 0.0
    return inner, (0.0, 0.0), s, 0.0
