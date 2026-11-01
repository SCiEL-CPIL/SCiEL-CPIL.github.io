import math

def construct():
    # Centers of the 7-hexagon flower
    sqrt3 = math.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0), (-sqrt3, 0.0),
        (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5), (-sqrt3/2, -1.5)
    ]
    
    # Add 4 hexagons at the L2 edge midpoints along the X-axis
    # These are at x = +/- 3*sqrt(3)/2, y = +/- 1.5
    # We choose the ones that extend the width in X
    # (3*sqrt(3)/2, 1.5) and (3*sqrt(3)/2, -1.5) extend right
    # (-3*sqrt(3)/2, 1.5) and (-3*sqrt(3)/2, -1.5) extend left
    
    new_centers = [
        (3*sqrt3/2, 1.5),
        (3*sqrt3/2, -1.5),
        (-3*sqrt3/2, 1.5),
        (-3*sqrt3/2, -1.5)
    ]
    
    centers.extend(new_centers)
    
    # Calculate bounding box
    max_x = max(c[0] for c in centers) + 1.0
    min_x = min(c[0] for c in centers) - 1.0
    max_y = max(c[1] for c in centers) + sqrt3/2
    min_y = min(c[1] for c in centers) - sqrt3/2
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Container constraints for aligned hexagon
    # Width across vertices: 2 * S
    # Height across sides: S * sqrt(3)
    S_from_width = width / 2.0
    S_from_height = height / sqrt3
    
    S = max(S_from_width, S_from_height)
    
    # Format inner hexagons: [x, y, angle_degrees]
    inner = [[x, y, 0.0] for x, y in centers]
    
    return inner, (0.0, 0.0), S, 0.0
