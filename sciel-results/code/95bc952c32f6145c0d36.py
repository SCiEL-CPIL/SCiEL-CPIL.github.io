import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for a hexagonal cluster
    # Ring 0: 1 hex
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes (we only need 4)
    
    centers = []
    
    # Ring 0
    centers.append((0.0, 0.0))
    
    # Ring 1
    # Axial coordinates (q, r)
    ring1_coords = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for q, r in ring1_coords:
        # Convert axial to cartesian for flat-top hexagons
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ring 2
    # We need 4 more.
    # Pick 4 from Ring 2.
    # Ring 2 axial coords:
    ring2_coords = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2)
    ]
    
    # Take the first 4
    for q, r in ring2_coords[:4]:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ensure we have exactly N centers
    # If N is smaller, we might have added too many? 
    # N=11. 1 + 6 + 4 = 11. Correct.
    
    inner = [[x, y, 0.0] for x, y in centers[:N]]
    
    # Calculate optimal outer side length
    # S = 2 * sqrt(3) + 1
    side = 2.0 * math.sqrt(3) + 1.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
