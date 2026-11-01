import math

def construct():
    # N = 11
    # We use a lattice-aligned packing.
    # 1 center, 6 neighbors (layer 1), 4 corner extensions (layer 2).
    
    # Lattice parameters for unit hexagons
    # Distance between centers of touching hexagons is sqrt(3)
    d = math.sqrt(3)
    
    # Center
    positions = [(0.0, 0.0)]
    
    # Layer 1: 6 neighbors
    # Angles 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle = math.radians(60 * i)
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        positions.append((x, y))
        
    # Layer 2: 4 corner positions
    # Corners are at distance 3 from center, angles 30, 90, 150, 210, 270, 330
    # We pick 4 of them to keep the shape somewhat balanced, though any 4 work for L=4.
    corner_angles = [30, 90, 150, 210]
    for deg in corner_angles:
        angle = math.radians(deg)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        positions.append((x, y))
        
    # Ensure we have exactly 11
    # 1 + 6 + 4 = 11
    
    # Format for output: [x, y, angle_degrees]
    # All inner hexagons are aligned (angle 0)
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Rotation 0.0
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
