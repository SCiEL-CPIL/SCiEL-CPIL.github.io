import math

N = 11

def construct():
    # 1 center hexagon
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 outer hexagons in a ring
    # Distance sqrt(3)
    d_ring = math.sqrt(3)
    for i in range(6):
        angle_rad = i * math.pi / 3
        x = d_ring * math.cos(angle_rad)
        y = d_ring * math.sin(angle_rad)
        # Angle 0 (vertex out)
        inner.append([x, y, 0.0])
        
    # 4 additional hexagons in the gaps (side midpoints)
    # Angles 30, 90, 150, 270 degrees
    # Distance d_gap
    # Constraint: d_gap >= 3 (to avoid collision with ring)
    # Constraint: d_gap <= sqrt(3)/2 * (S - 1)
    # We choose S such that these are equal: S = 1 + 2*sqrt(3)
    
    S = 1 + 2 * math.sqrt(3)
    d_gap = 3.0
    
    gap_angles = [30, 90, 150, 270]
    for deg in gap_angles:
        angle_rad = deg * math.pi / 180
        x = d_gap * math.cos(angle_rad)
        y = d_gap * math.sin(angle_rad)
        # Orient to fit? 
        # If we use angle 0, flat side normal is at 30.
        # The gap is at 30. So flat side faces the boundary.
        # This is optimal for fitting in the side gap.
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
