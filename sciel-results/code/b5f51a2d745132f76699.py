from fractions import Fraction
import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # We search for points with D=1 first.
    # Grid search
    # To keep it fast, we use a small grid.
    # If we find 6, we return.
    
    # Helper to check if a number is a perfect square of a rational
    def is_rational_square(val):
        if val < 0:
            return False
        # val is Fraction
        num = val.numerator
        den = val.denominator
        # Check if num and den are perfect squares
        sqrt_num = int(math.isqrt(num))
        sqrt_den = int(math.isqrt(den))
        if sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den:
            return True
        return False

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Check if determinant is 0
        # | x  y  x^2+y^2  1 |
        # | ... |
        # We can use the circle equation: x^2+y^2+Dx+Ey+F=0
        # Or simpler: check if the circumcircle of p1,p2,p3 passes through p4.
        # If p1,p2,p3 are collinear, they don't define a circle, but we already check collinearity for triples.
        # So we assume non-collinear.
        
        # Equation of circle through p1, p2, p3
        # x^2 + y^2 + Ax + By + C = 0
        # A, B, C can be found by solving linear system.
        # Or use determinant:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 | = 0
        # | x4 y4 x4^2+y4^2 1 |
        
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        # Using exact arithmetic
        # det = 0
        # To avoid huge numbers, we can use the property that 4 points are concyclic iff
        # (x1-x2)^2 + (y1-y2)^2 ... no.
        
        # Let's compute the circle coefficients A, B, C for p1, p2, p3.
        # x1^2+y1^2 + Ax1 + By1 + C = 0
        # x2^2+y2^2 + Ax2 + By2 + C = 0
        # x3^2+y3^2 + Ax3 + By3 + C = 0
        
        # Subtracts:
        # (x2^2-x1^2) + (y2^2-y1^2) + A(x2-x1) + B(y2-y1) = 0
        # (x3^2-x1^2) + (y3^2-y1^2) + A(x3-x1) + B(y3-y1) = 0
        
        # This is a 2x2 system for A, B.
        # Then check p4.
        
        # If p1,p2,p3 are collinear, return False (or True? No, collinear points are not concyclic in the strict sense of a circle, but the problem forbids 3 collinear anyway).
        
        dx12 = x2 - x1
        dy12 = y2 - y1
        dx13 = x3 - x1
        dy13 = y3 - y1
        
        # If dx12==0 and dy12==0, points are same, not allowed.
        # If dx12*dy13 - dx13*dy12 == 0, collinear.
        if dx12*dy13 - dx13*dy12 == 0:
            return False # Not a valid circle, but we check collinearity separately
        
        # Matrix:
        # [dx12, dy12] [A]   [-(x2^2-x1^2 + y2^2-y1^2)]
        # [dx13, dy13] [B] = [-(x3^2-x1^2 + y3^2-y1^2)]
        
        det = dx12*dy13 - dx13*dy12
        if det == 0:
            return False
            
        A = (-(x2*x2-x1*x1 + y2*y2-y1*y1)*dy13 + (-(x3*x3-x1*x1 + y3*y3-y1*y1))*dy12) / det
        # Wait, Cramer's rule:
        # A = (b1*c2 - b2*c1) / det ? No.
        # System:
        # dx12*A + dy12*B = c1
        # dx13*A + dy13*B = c2
        # c1 = -(x2^2-x1^2 + y2^2-y1^2)
        # c2 = -(x3^2-x1^2 + y3^2-y1^2)
        
        c1 = -(x2*x2-x1*x1 + y2*y2-y1*y1)
        c2 = -(x3*x3-x1*x1 + y3*y3-y1*y1)
        
        A = (c1*dy13 - c2*dy12) / det
        B = (dx12*c2 - dx13*c1) / det
        
        # C = -(x1^2+y1^2 + Ax1 + By1)
        C = -(x1*x1 + y1*y1 + A*x1 + B*y1)
        
        # Check p4:
        # x4^2 + y4^2 + A*x4 + B*y4 + C == 0
        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
        return val == 0

    # Grid of points
    # We use integers for simplicity, D=1
    # Range [-10, 10]
    points = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            points.append((Fraction(x), Fraction(y)))
            
    # Precompute valid edges
    # This might be memory heavy if we store all, but 441 points is small.
    valid_pairs = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d2 = dist_sq(points[i], points[j])
            if is_rational_square(d2):
                if i not in valid_pairs:
                    valid_pairs[i] = set()
                if j not in valid_pairs:
                    valid_pairs[j] = set()
                valid_pairs[i].add(j)
                valid_pairs[j].add(i)
                
    # Greedy search for clique of size 6
    best_set = []
    
    # Try to find a set of 6
    # We can iterate over all starting points
    # To save time, we just try a few random subsets or greedy from each point
    
    import random
    random.seed(42)
    
    # We want to maximize the set size.
    # We will try to build a set greedily.
    
    def try_build(start_idx, current_set, current_indices):
        # current_set is list of point indices
        if len(current_set) == 6:
            return current_set[:]
            
        if time.time() > deadline:
            return None
            
        # Candidates must be connected to all in current_set
        candidates = []
        for i in range(len(points)):
            if i in current_indices:
                continue
            ok = True
            for j in current_indices:
                if i not in valid_pairs.get(j, set()):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check general position
            # No 3 collinear
            collinear = False
            for k in range(len(current_set)-1):
                if is_collinear(points[i], points[current_set[k]], points[current_set[k+1]]):
                    collinear = True
                    break
            if collinear:
                continue
                
            # No 4 concyclic
            concyclic = False
            if len(current_set) >= 3:
                # Check if i is concyclic with any triple in current_set
                # We only need to check triples that don't include i (obviously)
                # Since we add i, we check if i is on the circle of any 3 existing points.
                for triple in itertools.combinations(current_set, 3):
                    if is_concyclic(points[i], points[triple[0]], points[triple[1]], points[triple[2]]):
                        concyclic = True
                        break
            if concyclic:
                continue
                
            candidates.append(i)
            
        # If no candidates, return None
        if not candidates:
            return None
            
        # Try each candidate
        # To improve search, we can sort candidates by degree or something
        for cand in candidates:
            res = try_build(cand, current_set + [cand], current_indices | {cand})
            if res is not None:
                return res
                
        return None

    # Try starting from each point
    found = None
    # To speed up, we might not try all 441 starting points if it's slow.
    # But 441 is small.
    # However, the recursion might be deep.
    # Let's limit the search.
    
    # We will try a subset of starting points first, then expand if needed.
    # Actually, just try all, but with a time check inside.
    
    for i in range(len(points)):
        if time.time() > deadline:
            break
        res = try_build(i, [i], {i})
        if res is not None:
            found = res
            break
            
    if found:
        # Convert to required format
        pts = [points[i] for i in found]
        return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]}
    else:
        # Fallback to seed
        points_seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points_seed]}
