import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    centers = []
    angles = []
    
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    
    # Ring 1: 6 hexagons
    d1 = math.sqrt(3)
    for i in range(6):
        theta = i * math.pi / 3
        x = d1 * math.cos(theta)
        y = d1 * math.sin(theta)
        centers.append((x, y))
        # Rotate inner hex to point outwards
        angles.append(math.degrees(theta))
        
    # Ring 2: 4 hexagons at corners
    d2 = 3.0
    corner_indices = [0, 1, 2, 3] # 30, 90, 150, 210
    # Actually, let's just use the angles directly
    corner_angles = [
        math.radians(30),
        math.radians(90),
        math.radians(150),
        math.radians(210)
    ]
    for theta in corner_angles:
        x = d2 * math.cos(theta)
        y = d2 * math.sin(theta)
        centers.append((x, y))
        angles.append(math.degrees(theta))
        
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    # Outer hexagon
    # Must contain vertices at distance 4 at angles 30, 90, 150, 210.
    # If outer hex has vertices at these angles, L=4 works.
    # Vertices of outer hex are at angle_offset + k*60.
    # We need 30, 90, 150, 210 to be in {30, 90, 150, 210, 270, 330}.
    # So offset = 30 degrees.
    
    return inner, (0.0, 0.0), 4.0, 30.0
# EVOLVE-BLOCK-END
