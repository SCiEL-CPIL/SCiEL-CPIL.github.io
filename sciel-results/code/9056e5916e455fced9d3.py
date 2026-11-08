import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3x4 grid in axial coordinates (q, r)
    # Flat-top hexagons
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    inner = []
    sqrt3 = math.sqrt(3)
    
    for r in range(3):
        for q in range(4):
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
            
    # Calculate bounding box of the cluster
    min_x = min(p[0] for p in inner) - 1.0
    max_x = max(p[0] for p in inner) + 1.0
    min_y = min(p[1] for p in inner) - sqrt3 / 2.0
    max_y = max(p[1] for p in inner) + sqrt3 / 2.0
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Outer hexagon (flat top) dimensions:
    # Width = 2 * S
    # Height = S * sqrt(3)
    
    S_from_width = width / 2.0
    S_from_height = height / sqrt3
    
    S = max(S_from_width, S_from_height) + 0.1 # Small margin
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    return inner, (center_x, center_y), S, 0.0
# EVOLVE-BLOCK-END
