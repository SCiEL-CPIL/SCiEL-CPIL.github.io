import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagon geometry
    # Unit hexagon side 1.
    # Lattice spacing sqrt(3).
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # Shell 1: 6 hexagons
    for i in range(6):
        angle = math.radians(60 * i)
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # Shell 2: We need 4 more.
    # Positions at distance 3, angles 30, 90, 150, 210, 270, 330.
    # We pick 4. To maintain symmetry and minimize bounding box,
    # picking any 4 that are spread out is good.
    # Let's pick angles 30, 90, 270, 330 (leaving 150, 210).
    # This leaves a gap on the left.
    # Alternatively, pick 0, 120, 240? No, lattice is 30, 90...
    # Let's pick 30, 90, 210, 270?
    # Actually, for a hexagon, symmetry is key.
    # If we pick 4, we break 6-fold symmetry to 2-fold or 1-fold.
    # However, the "bottleneck" distance 4.464 was derived assuming
    # that the "flat" side of the shell-2 hexagon is the limiting factor
    # in the edge direction.
    # This applies to ANY shell-2 hexagon.
    # So the L is determined by the presence of ANY shell-2 hexagon.
    
    shell2_angles = [30, 90, 150, 210, 270, 330]
    # Pick 4
    selected = [30, 90, 270, 330]
    
    for angle_deg in selected:
        angle = math.radians(angle_deg)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        centers.append((x, y))
        
    # Total 1 + 6 + 4 = 11 centers.
    
    # Rotation of inner hexagons
    # We found that angle 0 or 30 gives the same L.
    # Let's use angle 0 for simplicity.
    inner_angle = 0.0
    
    inner = []
    for (x, y) in centers:
        inner.append([x, y, inner_angle])
        
    # Outer hexagon
    # L = 4.4641016...
    # L = 3 + sqrt(3)/2 * (2/sqrt(3)) ?
    # L * sqrt(3)/2 = 3 + sqrt(3)/2
    # L = (3 + sqrt(3)/2) * 2 / sqrt(3)
    # L = 6/sqrt(3) + 1
    # L = 2*sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
# EVOLVE-BLOCK-END
