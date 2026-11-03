import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate axial coordinates for a hexagon of side 3 (19 hexes)
    # Axial coords (q, r) such that |q| <= 2, |r| <= 2, |q+r| <= 2
    hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                hexes.append((q, r))
    
    # Identify corners: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    corners = [(2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2)]
    
    # Remove corners
    remaining = [h for h in hexes if h not in corners]
    # remaining has 13 hexes
    
    # Take first 12
    selected = remaining[:12]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    inner = []
    for q, r in selected:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Calculate required side length S
    # Outer hexagon centered at (0,0), angle 0
    # Vertices of inner hex at (cx, cy) are (cx + cos(t), cy + sin(t))
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    
    max_s = 0.0
    for x, y, _ in inner:
        # Vertices of this inner hex
        vx = x + np.cos(angles)
        vy = y + np.sin(angles)
        
        # Check constraints for each vertex
        # 1. |vx| <= S
        s1 = np.max(np.abs(vx))
        
        # 2. |vy| <= sqrt(3)/2 * S  => S >= 2*|vy|/sqrt(3)
        s2 = np.max(2.0 * np.abs(vy) / np.sqrt(3))
        
        # 3. |sqrt(3)/2 * vx + 1/2 * vy| <= sqrt(3)/2 * S => S >= 2/ sqrt(3) * |...|
        proj3 = np.sqrt(3)/2.0 * vx + 0.5 * vy
        s3 = np.max(2.0 * np.abs(proj3) / np.sqrt(3))
        
        # 4. |sqrt(3)/2 * vx - 1/2 * vy| <= sqrt(3)/2 * S => S >= 2/ sqrt(3) * |...|
        proj4 = np.sqrt(3)/2.0 * vx - 0.5 * vy
        s4 = np.max(2.0 * np.abs(proj4) / np.sqrt(3))
        
        local_s = max(s1, s2, s3, s4)
        if local_s > max_s:
            max_s = local_s
            
    return inner, (0.0, 0.0), max_s, 0.0
# EVOLVE-BLOCK-END
