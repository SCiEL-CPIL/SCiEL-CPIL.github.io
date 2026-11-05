# EVOLVE-BLOCK-START
def construct():
    # Pre-computed optimal configuration for n=32 achieving Harborth's bound of 76 edges.
    # Hexagonal cluster of radius 3 (37 points) minus 5 boundary points.
    # The 32 points are all (a,b) with max(|a|,|b|,|a+b|) <= 3, except the 5 removed.
    skip = {(3, 0), (3, -1), (3, -2), (3, -3), (2, -3)}
    
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                if (a, b) not in skip:
                    pts.append([a, b])
    
    return pts
# EVOLVE-BLOCK-END
