import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Outer hexagon parameters
    # S = 3 + sqrt(3)/2
    side = 3.0 + math.sqrt(3.0) / 2.0
    outer_center = (0.0, 0.0)
    outer_angle = 30.0  # Degrees

    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. First ring: 6 hexagons at distance sqrt(3)
    dist1 = math.sqrt(3.0)
    for i in range(6):
        angle_deg = i * 60.0
        angle_rad = math.radians(angle_deg)
        x = dist1 * math.cos(angle_rad)
        y = dist1 * math.sin(angle_rad)
        # Orient with vertex pointing radially out (Angle 0 relative to radial)
        # Since global angle is angle_deg, and we want vertex at angle_deg,
        # and standard hex angle 0 has vertex at 0, we set angle to angle_deg.
        inner.append([x, y, angle_deg])
        
    # 3. Second ring: 4 hexagons at distance 3
    # Positions at 30, 90, 210, 270 degrees
    dist2 = 3.0
    target_angles = [30.0, 90.0, 210.0, 270.0]
    for angle_deg in target_angles:
        angle_rad = math.radians(angle_deg)
        x = dist2 * math.cos(angle_rad)
        y = dist2 * math.sin(angle_rad)
        # Orient with flat side pointing radially out.
        # Standard hex (angle 0) has flat side normal at 30.
        # We want flat side normal at angle_deg.
        # So we need to rotate by angle_deg - 30.
        rot_angle = angle_deg - 30.0
        inner.append([x, y, rot_angle])

    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
