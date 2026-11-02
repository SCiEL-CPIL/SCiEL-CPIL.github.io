import math

N = 12

def construct():
    # Generate 19 hexagon centers in a honeycomb cluster
    # Axial coordinates (q, r) to Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if within hexagon of radius 2 (side 3)
            # Condition: max(|q|, |r|, |q+r|) <= 2
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                centers.append((x, y))
    
    # We have 19 centers. We need 12.
    # Sort by distance from origin to pick the most central 12
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected = centers[:N]
    
    # Inner hexagons: (x, y, angle)
    # All aligned with angle 0
    inner = [[x, y, 0.0] for x, y in selected]
    
    # Outer hexagon
    # Side length S = 2*sqrt(3) + 1
    S = 2 * math.sqrt(3) + 1.0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, S, angle
