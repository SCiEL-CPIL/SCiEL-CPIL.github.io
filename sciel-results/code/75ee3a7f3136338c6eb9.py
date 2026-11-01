import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate the optimal side length for the outer hexagon
    # We use a rotated outer hexagon (30 deg) to fit the "closer" ring of lattice points
    # Lattice points at distance 3.0 from center (in physical units)
    # Outer hexagon vertices must be at least 3.0 away in those directions
    # If rotated 30 deg, vertices are at 30, 90, etc.
    # Distance from center to vertex is S.
    # Wait, inradius r = S * sqrt(3)/2.
    # If vertices are at 30 deg, the distance to the vertex is S.
    # The points are at 30 deg. So we need S >= 3.0.
    # But we also need to contain the hexagons themselves.
    # The "eroded" hexagon side L must satisfy:
    # In the direction of the vertex (30 deg), the distance to boundary is L (if flat-top? No).
    # Let's stick to the erosion logic.
    # Eroded hexagon side L.
    # If we rotate the outer hexagon by 30 deg, the eroded hexagon is also rotated 30 deg.
    # The vertices of the eroded hexagon are at distance L from center.
    # The points at dist 3.0 are at angles 30, 90, etc.
    # These align with the vertices of the rotated eroded hexagon.
    # So we need L >= 3.0.
    # S = L + 1 = 4.0?
    
    # Let's re-verify the rotation gain.
    # Unrotated: Vertices at 0, 60. Sides at 30, 90.
    # Points at 30 deg (dist 3.0) hit the side.
    # Distance to side is L * sqrt(3)/2.
    # Need L * sqrt(3)/2 >= 3.0 => L >= 3.0 * 2 / sqrt(3) = 2 * sqrt(3) = 3.464.
    # S = 4.464.
    
    # Rotated 30 deg: Vertices at 30, 90. Sides at 0, 60.
    # Points at 30 deg (dist 3.0) hit the vertex.
    # Distance to vertex is L.
    # Need L >= 3.0.
    # S = 4.0.
    
    # Can we do better?
    # What if we pick points from the dist 3.46 ring?
    # They are at 0, 60 deg.
    # In rotated frame, these are at -30, 30 relative to sides?
    # Distance to side is L.
    # Need L >= 3.464.
    # S = 4.464.
    
    # So rotating by 30 deg allows us to use the "dist 3.0" points, reducing S to 4.0.
    
    side_length = 4.0
    outer_angle = 30.0
    
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 points at distance sqrt(3)
    r1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Ring 2: 4 points at distance 3.0
    # The 6 points are at angles 30, 90, 150, 210, 270, 330
    r2 = 3.0
    for i in range(4):
        angle = math.radians(30 + 60 * i)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), side_length, outer_angle
# EVOLVE-BLOCK-END
