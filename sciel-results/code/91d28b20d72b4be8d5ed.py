"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Centers of 12 unit hexagons in a hexagonal lattice cluster
    # Core: 1 center + 6 neighbors
    # Outer: 5 from the next ring (choosing the ones with distance 3 from center to minimize bounding box)
    
    d1 = math.sqrt(3)
    d2 = 3.0 # Distance for "corner" positions of 2nd ring
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = math.radians(60 * i)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2 (5 hexagons)
    # Angles 30, 90, 150, 210, 270 (skipping 330)
    angles_r2 = [30, 90, 150, 210, 270]
    for deg in angles_r2:
        angle = math.radians(deg)
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        centers.append((x, y))
        
    # Inner hexagons rotated by 30 degrees to align flat sides with the container's flat sides (minimizing radial extent in y and x directions relative to container vertices/sides)
    # Container is aligned with angle 0 (vertices at 0, 60...). 
    # Inner hexagons at 30 deg have vertices at 30, 90... 
    # Wait, if container has flat top (normal 90), we want inner hex to have flat top (normal 90).
    # Standard hex (0 deg) has normal 30. Rotate by 60 -> normal 90.
    # Let's use 60 degrees for inner hexagons.
    inner_angle = 60.0
    
    inner = [[x, y, inner_angle] for x, y in centers]
    
    # Container
    # L = 4.5 should be safe. 
    # Max y of inner: 3.0 (center) + apothem (0.866) = 3.866.
    # Container max y: 4.5 * sqrt(3)/2 = 3.897. OK.
    # Max x of inner: 2.598 (center) + apothem (0.866) = 3.464.
    # Container max x (vertex): 4.5. OK.
    
    return inner, (0.0, 0.0), 4.5, 0.0
# EVOLVE-BLOCK-END
