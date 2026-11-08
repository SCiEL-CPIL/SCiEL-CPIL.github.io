import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Packing 11 hexagons into a regular hexagon
    # Using a grid-like arrangement
    inner = []
    
    # Row 0: 4 hexagons
    for i in range(4):
        inner.append([i * 1.732 - 2.598, 0.0, 0.0])
        
    # Row 1: 3 hexagons
    for i in range(3):
        inner.append([i * 1.732 - 1.732, 1.5, 0.0])
        
    # Row 2: 4 hexagons
    for i in range(4):
        inner.append([i * 1.732 - 2.598, -1.5, 0.0])
        
    # This packing requires S >= 4.2 to be safe
    return inner, (0.0, 0.0), 4.2, 0.0
# EVOLVE-BLOCK-END
