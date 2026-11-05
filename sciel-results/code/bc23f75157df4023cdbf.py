import time

# EVOLVE-BLOCK-START
def construct():
    # Pre-computed optimal configuration for n=32 achieving Harborth's bound of 76 edges.
    # Hexagonal cluster of radius 3 (37 points) minus 5 boundary points forming a path.
    # This attains the theoretical maximum floor(3*32 - sqrt(12*32-3)) = 76 unit-distance pairs.
    
    # The 32 points are all (a,b) with max(|a|,|b|,|a+b|) <= 3, except the 5 removed.
    remove_set = {
        (3, 0),
        (3, -1),
        (3, -2),
        (3, -3),
        (2, -3)
    }
    
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                if (a, b) not in remove_set:
                    points.append((a, b))
    
    # Verify we have exactly 32 distinct points
    if len(points) != 32:
        raise ValueError(f"Expected 32 points, got {len(points)}")
    
    # Verify all points are distinct and within bounds
    point_set = set(points)
    if len(point_set) != 32:
        raise ValueError("Points are not distinct")
    for a, b in points:
        if abs(a) > 64 or abs(b) > 64:
            raise ValueError(f"Point ({a},{b}) out of bounds")
    
    # Verify minimum distance constraint: dist_sq = da^2 + db^2 + da*db >= 1 for all pairs
    for i in range(32):
        for j in range(i + 1, 32):
            da = points[i][0] - points[j][0]
            db = points[i][1] - points[j][1]
            d2 = da * da + db * db + da * db
            if d2 < 1:
                raise ValueError(f"Points {points[i]} and {points[j]} too close: d2={d2}")
    
    return [[p[0], p[1]] for p in points]
# EVOLVE-BLOCK-END
