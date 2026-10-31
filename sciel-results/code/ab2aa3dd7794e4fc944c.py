import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Coordinates for 11 unit hexagons in a compact cluster
    # Using axial coordinates (q, r) to Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # 1 center
    centers = [(0, 0)]
    
    # 6 neighbors (ring 1)
    neighbors_ring1 = [
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1)
    ]
    for q, r in neighbors_ring1:
        centers.append((q, r))
        
    # 4 more from ring 2 (distance 3 from center)
    # These are at (1,1), (-2,1), (-1,-1), (2,-1)
    # This creates a symmetric 4-point extension
    extra_ring2 = [
        (1, 1), (-2, 1), (-1, -1), (2, -1)
    ]
    for q, r in extra_ring2:
        centers.append((q, r))
        
    # Convert to Cartesian and create inner list
    inner = []
    for q, r in centers:
        x, y = axial_to_cartesian(q, r)
        # Angle 0.0 for all inner hexagons
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # The cluster extends to distance 4.0 from the center.
    # A regular hexagon with side length L has circumradius L.
    # So L = 4.0 is sufficient.
    outer_center = (0.0, 0.0)
    outer_side = 4.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
