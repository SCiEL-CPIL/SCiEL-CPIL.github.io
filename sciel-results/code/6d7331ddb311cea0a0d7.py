import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Calculate the optimal side length
    # S = 2*sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1
    
    # Outer hexagon center and rotation
    # We determined that rotating the outer hexagon by 30 degrees
    # aligns its flats with the "wide" lobes of the inner cluster.
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    inner = []
    
    # 1. Central hexagon
    # Position (0,0), angle 0
    inner.append([0.0, 0.0, 0.0])
    
    # 2. First ring: 6 hexagons
    # Distance sqrt(3)
    dist1 = math.sqrt(3)
    for i in range(6):
        angle_rad = math.radians(60 * i)
        x = dist1 * math.cos(angle_rad)
        y = dist1 * math.sin(angle_rad)
        # Rotate to face radially outward
        angle_deg = 60 * i
        inner.append([x, y, angle_deg])
        
    # 3. Second ring: 5 hexagons
    # We pick the 5 "side" positions out of 6.
    # Side positions are at angles 30, 90, 150, 210, 270, 330.
    # We exclude 330 (which is -30) to align with the outer hexagon vertex.
    dist2 = 3.0
    angles_to_use = [30, 90, 150, 210, 270]
    for angle_deg in angles_to_use:
        angle_rad = math.radians(angle_deg)
        x = dist2 * math.cos(angle_rad)
        y = dist2 * math.sin(angle_rad)
        # Rotate to face radially outward
        inner.append([x, y, angle_deg])
        
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
