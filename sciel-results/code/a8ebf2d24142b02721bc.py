"""Rewritten program for the suite task."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    """
    Returns exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega, omega = exp(i*pi/3), which are at
    mutual distance at least one. Maximizes the number of pairs at distance
    exactly one; Harborth's theorem gives the optimum floor(3n - sqrt(12n - 3)) = 76.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Distance squared in the triangular lattice metric
    def dist2(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1 (dist2 == 1)."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist2(points[i], points[j]) == 1:
                    count += 1
        return count
    
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
    
    # Build the base 19-point hexagon (rings 0, 1, 2)
    base_points = [(0, 0)]
    for p in hex_ring(1):
        base_points.append(p)
    for p in hex_ring(2):
        base_points.append(p)
    
    # We need 13 more points from ring 3
    ring3_points = hex_ring(3)
    all_candidates = base_points + ring3_points
    
    # Precompute which ring-3 points are "compatible" with the base
    # (all points in the full hexagon are mutually at distance >= 1)
    
    # Try all C(18, 13) subsets of ring-3 points
    # C(18,13) = C(18,5) = 8568 - manageable
    best_edges = 0
    best_subset = None
    
    ring3_indices = list(range(18))
    
    for combo in combinations(ring3_indices, 13):
        if time.time() > deadline:
            break
        
        selected_ring3 = [ring3_points[i] for i in combo]
        candidate = base_points + selected_ring3
        
        # Verify all pairs at distance >= 1
        valid = True
        for i in range(len(candidate)):
            for j in range(i+1, len(candidate)):
                if dist2(candidate[i], candidate[j]) < 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        edges = count_edges(candidate)
        if edges > best_edges:
            best_edges = edges
            best_subset = combo
            if best_edges == 76:
                break
    
    # If we found 76, use it
    if best_edges == 76:
        selected_ring3 = [ring3_points[i] for i in best_subset]
        result = base_points + selected_ring3
    else:
        # Fallback: use the first 13 ring-3 points
        result = base_points + ring3_points[:13]
    
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
