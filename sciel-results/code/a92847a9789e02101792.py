"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                det = abs(xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Strategy: 28 collinear points on y=0, 2 points at y=1
    # Collinear points: (i, 0) for i in range(28) -> x from -14 to 13
    # Off-line points: (0, 1) and (1, 1)
    # Doubled area for triangle with 2 collinear pts (a,0),(b,0) and 1 off-line pt (c,1):
    #   det = |a*(0-1) + b*(1-0) + c*(0-0)| = |b - a| = |a-b|
    # So for each off-line point, we get areas = all |a-b| for pairs of collinear points
    # With 28 collinear points at x=0..27, differences 1..27 with multiplicities 27,26,...,1
    # Total triangles of area d: 2 * (28 - d) for d in 1..27
    # Max multiplicity: 2 * 27 = 54 for area 1
    
    # Better: use 29 collinear + 1 off-line
    # 29 collinear at x=-14..14, 1 off-line at (0,1)
    # Differences 1..28, multiplicity of d is (29-d)
    # Max: 29-1 = 28 for area 1... wait that's only 28 triangles
    # C(29,2) = 406 total triangles, spread over 28 area values
    # Area 1: 28 triangles, area 2: 27, ..., area 28: 1
    # Max multiplicity = 28
    
    # Even better: 28 collinear + 2 off-line at same y
    # 28 collinear at x=0..27, off-line at (0,1) and (27,1)
    # For off-line at (0,1): area = |b-a| for each pair (a,b)
    # For off-line at (27,1): area = |b-a| for each pair (a,b)
    # Total for area d: 2*(28-d)
    # Max at d=1: 2*27 = 54
    
    # Try 27 collinear + 3 off-line
    # 27 collinear at x=0..26, 3 off-line at y=1
    # For each off-line: area = |b-a|, multiplicity (27-d)
    # Total: 3*(27-d), max at d=1: 3*26 = 78
    
    # Try 26 collinear + 4 off-line
    # 4*(26-1) = 100
    
    # Try 25 collinear + 5 off-line
    # 5*(25-1) = 120
    
    # Try 24 collinear + 6 off-line
    # 6*(24-1) = 138
    
    # Try 23 collinear + 7 off-line
    # 7*(23-1) = 154
    
    # Try 22 collinear + 8 off-line
    # 8*(22-1) = 168
    
    # Try 21 collinear + 9 off-line
    # 9*(21-1) = 180
    
    # Try 20 collinear + 10 off-line
    # 10*(20-1) = 190
    
    # Try 19 collinear + 11 off-line
    # 11*(19-1) = 198
    
    # Try 18 collinear + 12 off-line
    # 12*(18-1) = 204
    
    # Try 17 collinear + 13 off-line
    # 13*(17-1) = 208
    
    # Try 16 collinear + 14 off-line
    # 14*(16-1) = 210
    
    # Try 15 collinear + 15 off-line
    # 15*(15-1) = 210
    
    # Maximum around 16-17 collinear + 14-15 off-line: ~210
    
    # But wait, the off-line points also form triangles among themselves and with single collinear points
    # Those have different areas, so they don't contribute to the max multiplicity
    
    # Let's verify: with 16 collinear at x=0..15 on y=0, and 14 points at y=1
    # The 14 off-line points at y=1: if they're at various x, triangles with 2 off-line + 1 collinear
    # have area depending on the x-coordinates of the 2 off-line points
    # Triangles with 3 off-line points: area depends on their x-coordinates
    
    # The key insight: if all 14 off-line points are at the same y=1, and we want to maximize
    # the multiplicity of area 1, we need many pairs of collinear points with difference 1
    # That gives 15 pairs * 14 = 210 triangles of area 1
    
    # But we should also check if we can do better with a different configuration
    
    # Let's try: all 30 points on 2 parallel lines
    # Line 1: y=0, points at x = 0,1,...,14 (15 points)
    # Line 2: y=1, points at x = 0,1,...,14 (15 points)
    # Triangle with 2 on line 1 and 1 on line 2: area = |x1-x2|
    # Triangle with 2 on line 2 and 1 on line 1: area = |x1-x2|
    # Triangle with 1 on line 1 and 2 on line 2: area = |x1-x2|
    # Triangle with 3 on same line: area = 0 (excluded)
    # Triangle with 1 on line 1, 1 on line 2, and... wait, we need 3 points
    
    # For 2 on y=0 and 1 on y=1: doubled area = |x_a - x_b| where (x_a,0),(x_b,0) are the y=0 points
    # Number of such triangles with area d: C(15,2) pairs... no, for each pair (a,b) on y=0 with |a-b|=d,
    # and each point on y=1, we get one triangle. So 14 * (15-d) triangles of area d from this type.
    # Similarly for 2 on y=1 and 1 on y=0: 14 * (15-d)
    # Total for area d: 28 * (15-d)
    # Max at d=1: 28 * 14 = 392
    
    # That's much better! Let me verify the coordinates fit in [-30,30]^2
    # x from 0 to 14, y from 0 to 1: all in [-30,30]^2. 
    
    # Can we do even better? What about 3 parallel lines?
    # 10 points on y=0, 10 on y=1, 10 on y=2
    # 2 on y=0, 1 on y=1: area = |a-b|, count: 10*9/2... no, (10-d) pairs * 10 points on y=1
    # 2 on y=0, 1 on y=2: area = 2*|a-b|, count: (10-d) * 10
    # 2 on y=1, 1 on y=0: area = |a-b|, count: (10-d) * 10
    # 2 on y=1, 1 on y=2: area = |a-b|, count: (10-d) * 10
    # 2 on y=2, 1 on y=0: area = 2*|a-b|, count: (10-d) * 10
    # 2 on y=2, 1 on y=1: area = |a-b|, count: (10-d) * 10
    
    # For area 1: from (2 on y=0, 1 on y=1) + (2 on y=1, 1 on y=0) + (2 on y=1, 1 on y=2) + (2 on y=2, 1 on y=1)
    # = 4 * (10-1) * 10 = 4 * 9 * 10 = 360
    # That's less than 392.
    
    # So 2 lines with 15+15 is better. Let me try 16+14:
    # 2 on y=0 (16 pts), 1 on y=1 (14 pts): (16-d)*14
    # 2 on y=1 (14 pts), 1 on y=0 (16 pts): (14-d)*16
    # Total for area d: 14*(16-d) + 16*(14-d) = 14*16 - 14d + 16*14 - 16d = 224 - 30d
    # Max at d=1: 224 - 30 = 194. Worse than 392.
    
    # What about 20+10?
    # 2 on y=0 (20), 1 on y=1 (10): (20-d)*10
    # 2 on y=1 (10), 1 on y=0 (20): (10-d)*20
    # Total: 10*(20-d) + 20*(10-d) = 200 - 10d + 200 - 20d = 400 - 30d
    # Max at d=1: 370. Close to 392 but worse.
    
    # 15+15 gives 28*(15-1) = 28*14 = 392 for area 1.
    # Let me double-check: 15 points on each line.
    # Pairs on y=0 with difference 1: 14 pairs. Each with 15 points on y=1: 14*15 = 210 triangles.
    # Pairs on y=1 with difference 1: 14 pairs. Each with 15 points on y=0: 14*15 = 210 triangles.
    # Total: 420? Wait, I think I made an error.
    
    # Actually: for 2 points on y=0 and 1 point on y=1:
    # Number of pairs (a,b) on y=0 with |a-b|=d: (15-d)
    # For each such pair, 15 choices for the point on y=1
    # So (15-d)*15 triangles of area d from this type
    # Similarly (15-d)*15 from the other type
    # Total: 2*(15-d)*15 = 30*(15-d)
    # Max at d=1: 30*14 = 420
    
    # Let me recompute: 15 points at x=0,1,...,14
    # Pairs with difference 1: (0,1),(1,2),...,(13,14) = 14 pairs
    # Each pair + each of 15 points on other line: 14*15 = 210
    # Two types: 420 total for area 1.
    
    # Can we do better with more lines or different spacing?
    # What if the two lines are at y=0 and y=1, but with more points?
    # We're limited to 30 total. 15+15 seems optimal for 2 lines.
    
    # Let's also consider: can we get area 1 from other combinations?
    # 1 on y=0, 1 on y=1, 1 on... we only have 2 lines, so the third point is on one of the lines.
    # That's already counted.
    
    # So the answer should be 420 for the 15+15 configuration.
    
    # Let me construct it:
    points = []
    for x in range(15):
        points.append((x, 0))
    for x in range(15):
        points.append((x, 1))
    
    # Verify all distinct and in range
    assert len(set(points)) == 30
    assert all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in points)
    
    score = _score(points)
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 5:
        import random
        best_points = list(points)
        best_score = score
        random.seed(42)
        
        while time.time() < deadline - 2:
            # Try swapping a point
            new_points = list(best_points)
            idx = random.randint(0, 29)
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_points[idx] = (new_x, new_y)
            
            # Check distinctness
            if len(set(new_points)) != 30:
                continue
            
            s = _score(new_points)
            if s > best_score:
                best_score = s
                best_points = new_points
    
    return best_points
# EVOLVE-BLOCK-END

