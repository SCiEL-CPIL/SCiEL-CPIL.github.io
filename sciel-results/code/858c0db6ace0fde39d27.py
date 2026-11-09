import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 unit hexagons in a 3x4 grid (3 columns, 4 rows)
    # Flat-top orientation
    hex_width = math.sqrt(3)
    hex_height = 2.0
    
    centers = []
    for row in range(4):
        for col in range(3):
            x = col * hex_width
            y = row * 1.0  # Vertical offset for flat-top is 1? 
            # Flat top: Vertical distance between rows is 1?
            # Height is sqrt(3). Apothem is sqrt(3)/2.
            # Vertical spacing is 1?
            # Let's check.
            # Flat top hex:
            # Top edge y = sqrt(3)/2. Bottom edge y = -sqrt(3)/2.
            # Height sqrt(3).
            # Vertical distance between centers of adjacent rows:
            # If offset, distance is 1?
            # Let's use pointy-top for simpler grid?
            # Pointy top:
            # Width sqrt(3). Height 2.
            # Horizontal spacing 1.5. Vertical spacing sqrt(3).
            # Let's stick to flat-top.
            # Vertical spacing between rows: 1.0?
            # No, apothem is sqrt(3)/2.
            # Distance between parallel edges is sqrt(3).
            # If we stack them, vertical distance is 1?
            # Let's use the standard hex grid coords.
            pass
            
    # Let's use a simpler approach:
    # Place them in a hexagonal cluster of side 2 (7 hexes) and add 5.
    # Or just use the 3x4 block.
    
    # 3x4 Block Coordinates (Pointy Top)
    # Width = sqrt(3) * 3? No.
    # Let's use Pointy Top.
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Offset: sqrt(3)/2
    
    centers = []
    for r in range(4):
        for c in range(3):
            x = c * 1.5
            y = r * math.sqrt(3)
            if c % 2 == 1:
                y += math.sqrt(3) / 2
            centers.append((x, y, 0.0))
            
    # Center the cluster
    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    inner = [[x - cx, y - cy, 0.0] for x, y, _ in centers]
    
    # Calculate required side length
    # Max distance from center to any vertex
    max_dist = 0
    for x, y, _ in inner:
        # Vertices of unit hex (pointy top) relative to center
        # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
        for vx, vy in [(0, 1), (math.sqrt(3)/2, 0.5), (math.sqrt(3)/2, -0.5), (0, -1), (-math.sqrt(3)/2, -0.5), (-math.sqrt(3)/2, 0.5)]:
            dist = math.sqrt((x+vx)**2 + (y+vy)**2)
            if dist > max_dist:
                max_dist = dist
                
    side = max_dist
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
