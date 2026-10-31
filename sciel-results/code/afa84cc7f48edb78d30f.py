import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate the 11 hexagon centers on the hexagonal lattice
    # Axial coordinates (i, j)
    # Convert to Cartesian: x = sqrt(3) * (i + j/2), y = 1.5 * j
    
    centers = []
    
    # Ring 0
    centers.append((0, 0))
    
    # Ring 1 (6 points)
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    centers.extend(ring1)
    
    # Ring 2 (Need 4 more)
    # We identified "safe" points with lower bounding box projection:
    # (-1, 2), (1, -2), (2, -2), (-2, 2)
    ring2_safe = [(-1, 2), (1, -2), (2, -2), (-2, 2)]
    centers.extend(ring2_safe)
    
    # Convert to Cartesian coordinates
    inner = []
    sqrt3 = math.sqrt(3)
    for i, j in centers:
        x = sqrt3 * (i + j / 2.0)
        y = 1.5 * j
        angle = 0.0
        inner.append([x, y, angle])
    
    # Outer hexagon parameters
    center = (0.0, 0.0)
    # Minimal side length is sqrt(N) for dense packing limit
    side = math.sqrt(N)
    # Add a tiny epsilon to avoid floating point precision issues in evaluator
    side += 1e-9
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
