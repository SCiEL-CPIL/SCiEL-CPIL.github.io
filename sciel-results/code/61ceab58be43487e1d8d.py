import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in a compact cluster
    # 7 central hexes (K=2 cluster) + 4 outer hexes from K=3 layer
    # Axial coordinates (q, r)
    # K=2 cluster:
    # (0,0), (1,0), (-1,0), (0,1), (-1,1), (0,-1), (1,-1)
    # K=3 additions (corners):
    # (2,0), (-2,0), (0,2), (0,-2)
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (-1, 1),
        (0, -1), (1, -1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to cartesian for pointy-top hexes
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    # Calculate minimal side length
    # Max projection is 6 + sqrt(3)
    # S * sqrt(3) >= 6 + sqrt(3)
    # S >= 2*sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1.0 + 1e-6
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
