"""Rewritten program: minimize largest 3-collinear-free subset for 20 points."""
import time
import itertools

def construct():
    start_time = time.time()
    
    # 10 x-coordinates, each with 2 y-values -> 20 points
    # Design: use x = 1..10, and for each x, two y-values
    # The y-values are chosen to create cross-collinearities
    
    # Strategy: use a "grid-like" structure where points from different
    # x-columns align on shared lines, creating many collinear triples.
    
    # Let's use x in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    # For each x, assign two y-values from a structured pattern.
    
    # A good approach: use y-values that are linear functions of x
    # If we have points (x, a*x + b) for various a, b, then points
    # sharing the same (a,b) line are collinear.
    
    # With 20 points and 10 lines (each line having 2 points), 
    # we can create many collinear triples if lines intersect at points.
    
    # Actually, let's use a different approach:
    # Place points on a small number of lines, but ensure no 4 are on one line.
    # If we have 10 lines, each with 2 points, that's 20 points.
    # But we need no 4 collinear, so max 3 per line.
    
    # Better: use 5 lines with 4 points each? No, that violates "no 4 collinear".
    # So max 3 per line.
    
    # Let's use a configuration based on known small cap set constructions.
    
    # Here's a concrete design:
    # Use points on the parabola y = x^2 but perturb some to create collinear triples.
    # Actually, let's use a structured grid approach.
    
    # Design: 10 pairs at x = 1..10
    # For pair i at x=i, use y-values: y_i0 and y_i1
    # Choose y-values so that (i, y_i0), (j, y_j0), (k, y_k0) are collinear for many triples
    
    # Let's use y_i0 = i (diagonal) and y_i1 = 2*i (another diagonal)
    # Then points (i, i) for i=1..10 are all collinear (line y=x) - that's 10 collinear, too many!
    
    # We need no 4 collinear. So we can't have 10 points on one line.
    
    # New approach: use a "bent" configuration.
    # Place points so that collinear triples are abundant but no line has 4+.
    
    # Let me use a specific known construction idea:
    # Take a 5x4 grid and select 20 points with structure.
    
    # Actually, let's think about it differently.
    # We want to MINIMIZE the largest 3-collinear-free subset.
    # The minimum possible is related to how "collinear-dense" the set is.
    
    # If we can arrange that every subset of size k contains a collinear triple,
    # then the score is at most k-1.
    
    # For 20 points, if we can make it so that any 8 points contain a collinear triple,
    # the score would be 7. That's a strong target.
    
    # Let's try a construction with many 3-point lines.
    
    # Use points: for i in 0..9, j in 0..1:
    #   x = i + 1
    #   y = j * 10 + f(i)  where f creates collinearities
    
    # Simpler: let's use two "layers" of a modified parabola.
    
    # Layer 1: (i, i^2) for i in 0..9 -> 10 points, no 3 collinear
    # Layer 2: (i, i^2 + 1) for i in 0..9 -> 10 points
    # Check collinearities: 
    #   Within layer 1: no 3 collinear (parabola property)
    #   Within layer 2: no 3 collinear (shifted parabola)
    #   Across layers: (i, i^2), (j, j^2), (k, k^2+1) collinear?
    #   This might create some collinear triples.
    
    # Let's verify: are (0,0), (1,1), (2,5) collinear? 
    #   Slope 0->1: 1/1 = 1, slope 1->2: 4/1 = 4. No.
    #   (0,0), (1,1), (3,10): slope 0->1: 1, slope 1->3: 9/2=4.5. No.
    #   (0,0), (2,4), (1,2): slope 0->2: 2, slope 2->1: -2/1=-2. No.
    
    # Hmm, this might not create enough collinear triples.
    
    # Let me try a different structured approach.
    # Use points on lines y = mx + b for various m, b, with 2-3 points per line.
    
    # Concrete plan:
    # Create 10 lines, each containing exactly 2 of our points, 
    # but arrange so that many triples of points (from 3 different lines) are collinear.
    
    # Actually, the simplest strong construction:
    # Use a 4x5 grid subset with careful selection.
    
    # Let me just use a well-tested approach: 
    # Points on the moment curve with perturbations.
    
    # Final design: use 20 points where we have many 3-point collinear sets.
    
    # I'll use this specific configuration:
    # x-coords: 1,2,3,4,5,6,7,8,9,10 (each used twice)
    # y-coords for x=i: y = 2*i and y = 2*i + 1
    
    # So points are: (i, 2i) and (i, 2i+1) for i=1..10
    
    # Check: no 4 collinear?
    # Line through (i, 2i) and (j, 2j) has slope 2, equation y=2x. 
    # All 10 points (i, 2i) are on y=2x! That's 10 collinear. BAD.
    
    # Fix: use y = i^2 and y = i^2 + 1
    # Points: (i, i^2) and (i, i^2+1) for i=0..9
    
    # Within layer 1 (y=i^2): parabola, no 3 collinear. Good.
    # Within layer 2 (y=i^2+1): shifted parabola, no 3 collinear. Good.
    # Across: need to check for 4 collinear.
    # A line can intersect the parabola y=x^2 in at most 2 points, and y=x^2+1 in at most 2 points.
    # So a line can have at most 4 points total (2 from each parabola).
    # We need to verify no line actually has 4 points.
    
    # Let's check if there's a line with 2 points from y=x^2 and 2 from y=x^2+1.
    # Points (a, a^2), (b, b^2), (c, c^2+1), (d, d^2+1) collinear.
    # This is a system that may or may not have solutions in {0..9}.
    
    # For safety, let's use a larger spread and verify.
    
    points = []
    for i in range(10):
        x = i
        points.append([x, x * x])
        points.append([x, x * x + 1])
    
    # Verify: 20 distinct points, |x|,|y| <= 100
    assert len(points) == 20
    assert len(set(map(tuple, points))) == 20
    for p in points:
        assert abs(p[0]) <= 100 and abs(p[1]) <= 100
    
    # Check no 4 collinear
    def check_no_four_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        # Check if pts[i], pts[j], pts[k], pts[l] are collinear
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        # Cross product test
                        d1 = (p2[0]-p1[0]) * (p3[1]-p1[1]) - (p2[1]-p1[1]) * (p3[0]-p1[0])
                        d2 = (p2[0]-p1[0]) * (p4[1]-p1[1]) - (p2[1]-p1[1]) * (p4[0]-p1[0])
                        if d1 == 0 and d2 == 0:
                            return False
        return True
    
    if not check_no_four_collinear(points):
        # Fallback: use a safer configuration
        # Spread points more to avoid 4-collinear
        points = []
        for i in range(10):
            x = i * 2
            points.append([x, x * x // 4])
            points.append([x + 1, (x + 1) * (x + 1) // 4])
        
        # Verify again
        assert len(points) == 20
        assert len(set(map(tuple, points))) == 20
        for p in points:
            assert abs(p[0]) <= 100 and abs(p[1]) <= 100
        assert check_no_four_collinear(points), "Fallback also failed"
    
    # Check deadline
    if time.time() - start_time > 120:
        return points
    
    return points
