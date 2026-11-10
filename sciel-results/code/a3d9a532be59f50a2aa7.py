"""Rewritten program for the suite task."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Returns exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega, omega = exp(i*pi/3), which are at
    mutual distance at least one. Maximizes the number of pairs at distance
    exactly one; Harborth's theorem gives the optimum floor(3n - sqrt(12n - 3)) = 76.
    """
    # Distance squared in the triangular lattice metric
    def dist2(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def hex_ring(r):
        """Generate points on hexagonal ring r (r >= 1)."""
        if r == 0:
            return [(0, 0)]
        points = []
        corners = [(r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)]
        for i in range(6):
            c1 = corners[i]
            c2 = corners[(i+1) % 6]
            da = (c2[0] - c1[0]) // r
            db = (c2[1] - c1[1]) // r
            for step in range(r):
                points.append((c1[0] + da*step, c1[1] + db*step))
        return points

    # Generate the full hexagon of radius 3 (37 points)
    points = [(0, 0)]
    for r in range(1, 4):
        points.extend(hex_ring(r))
    
    # We need to remove 5 points to get to 32.
    # Strategy: Greedy removal of points with the fewest neighbors in the current set.
    # Ties are broken by removing points with larger distance from origin (keeping the core compact).
    
    current_points = set(points)
    
    for _ in range(5):
        # Calculate neighbor count for each point in current_points
        min_count = float('inf')
        min_point = None
        
        for p in current_points:
            count = 0
            for q in current_points:
                if p != q and dist2(p, q) == 1:
                    count += 1
            
            # Tie-breaking: prefer removing points further from origin
            # dist2(p, (0,0)) is p[0]^2 + p[0]p[1] + p[1]^2
            rad2 = p[0]*p[0] + p[0]*p[1] + p[1]*p[1]
            
            # We want to minimize count. If counts are equal, maximize rad2.
            # So we can sort by (count, -rad2) and pick the first.
            # Or just track the best candidate.
            if count < min_count or (count == min_count and min_point is not None and rad2 > (min_point[0]*min_point[0] + min_point[0]*min_point[1] + min_point[1]*min_point[1])):
                min_count = count
                min_point = p
        
        if min_point:
            current_points.remove(min_point)
    
    result = list(current_points)
    
    # Final verification
    assert len(result) == 32, f"Expected 32 points, got {len(result)}"
    assert len(set(result)) == 32, "Points are not distinct"
    for p in result:
        assert abs(p[0]) <= 64, f"Point {p} has |a| > 64"
        assert abs(p[1]) <= 64, f"Point {p} has |b| > 64"
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert dist2(result[i], result[j]) >= 1, f"Points {result[i]} and {result[j]} at distance < 1"
            
    return [[p[0], p[1]] for p in result]

# EVOLVE-BLOCK-END
