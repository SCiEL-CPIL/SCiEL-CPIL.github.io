import math

def construct():
    # Generate all lattice points within hexagonal distance 3 from origin
    # Hex distance d(a,b) = max(|a|, |b|, |a+b|)
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = max(abs(a), abs(b), abs(a+b))
            if d <= 3:
                # Calculate angle for sorting to ensure contiguous spiral
                # atan2 for triangular lattice coordinates
                # Convert to cartesian: x = a + b/2, y = b*sqrt(3)/2
                x = a + b / 2.0
                y = b * math.sqrt(3) / 2.0
                angle = math.atan2(y, x)
                points.append((d, angle, a, b))
    
    # Sort by distance, then by angle to create a spiral
    points.sort(key=lambda x: (x[0], x[1]))
    
    # Take the first 32 points
    selected = [ [a, b] for d, angle, a, b in points[:32] ]
    
    return selected
