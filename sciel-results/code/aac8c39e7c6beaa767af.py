import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice vectors for flat-top hexagons aligned with outer hexagon
    # Neighbors at distance sqrt(3)
    # Centers are defined by axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = sqrt(3)/2 * r
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 neighbors
    # Axial directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1_coords:
        x = math.sqrt(3) * (q + r/2.0)
        y = math.sqrt(3)/2.0 * r
        centers.append((x, y))
        
    # Ring 2: We need 4 more.
    # Corner points (dist 3) are at (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # Let's pick 4 of these. Due to symmetry of the outer hexagon (6-fold),
    # picking any 4 will result in the same max extent if the outer hexagon is aligned.
    # However, to be safe, we pick 4 that are "spread out" or just pick the first 4.
    # The max extent is determined by the max projection.
    # As analyzed, max extent is 4.
    # Let's pick (1,1), (-1,2), (-2,1), (2,-1) ? No, let's just pick 4 distinct ones.
    # (1,1) -> angle 30
    # (-1,2) -> x = sqrt(3)(-1+1)=0, y=sqrt(3). Angle 90.
    # (-2,1) -> x = sqrt(3)(-2+0.5) = -1.5sqrt(3), y=sqrt(3)/2. Angle 150.
    # (-1,-1) -> x = sqrt(3)(-1-0.5) = -1.5sqrt(3), y=-sqrt(3)/2. Angle 210.
    
    ring2_coords = [(1,1), (-1,2), (-2,1), (-1,-1)]
    for q, r in ring2_coords:
        x = math.sqrt(3) * (q + r/2.0)
        y = math.sqrt(3)/2.0 * r
        centers.append((x, y))
        
    # Construct inner list
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length S
    # Max extent of vertices is 4.
    # Apothem A = S * sqrt(3)/2
    # A >= 4
    # S >= 8 / sqrt(3)
    S = 8.0 / math.sqrt(3)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
