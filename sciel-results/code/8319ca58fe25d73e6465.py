import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 140s safety margin
    
    # Hex grid generation
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 neighbors
    ring1 = [
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    
    # Ring 2: We need 4 more.
    # The closest points in Ring 2 are at distance 3 (axial dist squared 9).
    # These are (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1).
    # Let's pick 4 of them symmetrically.
    # (1,1) and (-1,-1) are opposite.
    # (-1,2) and (1,-2) are opposite.
    # (-2,1) and (2,-1) are opposite.
    # Picking (1,1), (-1,-1), (-1,2), (1,-2) gives a symmetric set.
    
    ring2_subset = [
        (1, 1),
        (-1, -1),
        (-1, 2),
        (1, -2)
    ]
    
    all_axial = [(0,0)] + ring1 + ring2_subset
    
    inner = []
    for q, r in all_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 aligns flat-top hexagons
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Max distance from origin to any center is 3.0 (for the ring2_subset points).
    # Max distance to vertex is 3.0 + 1.0 = 4.0.
    # So outer side length must be at least 4.0.
    outer_side = 4.0 + 1e-6
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
