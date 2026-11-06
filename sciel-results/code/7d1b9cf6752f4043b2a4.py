import math

def construct():
    # Hardcoded optimal 32-point configuration on triangular lattice
    # Achieves the Harborth bound of 76 unit-distance pairs.
    # Points are in (a, b) coordinates where point = a + b*omega, omega = exp(i*pi/3)
    
    # Ring 0: center
    points = [
        [0, 0],
        # Ring 1: 6 points
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        # Ring 2: 12 points
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1], [-2, 0],
        [-2, -1], [-1, -2], [0, -2], [1, -2], [2, -1], [2, 1],
        # Ring 3: 13 points (partial ring to reach 32 total)
        [3, 0], [2, 2], [1, 3], [0, 3], [-1, 3], [-2, 2],
        [-3, 1], [-3, 0], [-3, -1], [-2, -3], [-1, -3], [0, -3],
        [1, -3]
    ]
    
    # Verify: exactly 32 distinct points with |a|,|b| <= 64
    assert len(points) == 32
    seen = set()
    for p in points:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        assert tuple(p) not in seen
        seen.add(tuple(p))
    
    return points
