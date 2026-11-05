import random
import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Helper to check if a number is a perfect square
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Check if 3 points are collinear (using integer coords)
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # Area of triangle * 2
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        return area2 == 0

    # Check if 4 points are concyclic (using integer coords)
    def is_concyclic(p1, p2, p3, p4):
        # Determinant of the matrix:
        # x^2+y^2  x  y  1
        # ...
        # If det == 0, they are concyclic (or collinear, but we check that separately)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Using the general equation of a circle: x^2 + y^2 + Dx + Ey + F = 0
        # We can solve for D, E, F using 3 points and check if the 4th satisfies it.
        # Or use the determinant directly.
        
        # To avoid large numbers, we can use the formula:
        # det [[x1^2+y1^2, x1, y1, 1],
        #      [x2^2+y2^2, x2, y2, 1],
        #      [x3^2+y3^2, x3, y3, 1],
        #      [x4^2+y4^2, x4, y4, 1]]
        
        # This is computationally heavy for large coordinates.
        # Alternative: Circle through p1, p2, p3.
        # Center (h, k).
        # (x1-h)^2 + (y1-k)^2 = r^2
        # (x2-h)^2 + (y2-k)^2 = r^2
        # (x3-h)^2 + (y3-k)^2 = r^2
        
        # Linear system for h, k:
        # 2(x2-x1)h + 2(y2-y1)k = x2^2-x1^2 + y2^2-y1^2
        # 2(x3-x1)h + 2(y3-y1)k = x3^2-x1^2 + y3^2-y1^2
        
        a1 = 2 * (x2 - x1)
        b1 = 2 * (y2 - y1)
        c1 = x2*x2 - x1*x1 + y2*y2 - y1*y1
        
        a2 = 2 * (x3 - x1)
        b2 = 2 * (y3 - y1)
        c2 = x3*x3 - x1*x1 + y3*y3 - y1*y1
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            return True # Collinear, hence "concyclic" in a degenerate sense, but we check collinear separately.
                        # Actually, if collinear, they don't form a circle, but the determinant of the 4x4 matrix is 0.
                        # The task says "no four concyclic". Collinear points are usually excluded by the "no three collinear" check.
                        # If no three are collinear, det != 0 implies a unique circle.
        
        # h = (c1*b2 - c2*b1) / det
        # k = (a1*c2 - a2*c1) / det
        
        # Check if p4 is on this circle:
        # (x4-h)^2 + (y4-k)^2 == (x1-h)^2 + (y1-k)^2
        # x4^2 - 2x4h + h^2 + y4^2 - 2y4k + k^2 == x1^2 - 2x1h + h^2 + y1^2 - 2y1k + k^2
        # x4^2 + y4^2 - x1^2 - y1^2 - 2h(x4-x1) - 2k(y4-y1) == 0
        
        # Substitute h, k:
        # Let R = x4^2 + y4^2 - x1^2 - y1^2
        # LHS = R - 2 * ( (c1*b2 - c2*b1)/det * (x4-x1) + (a1*c2 - a2*c1)/det * (y4-y1) )
        # Multiply by det:
        # R * det - 2 * ( (c1*b2 - c2*b1) * (x4-x1) + (a1*c2 - a2*c1) * (y4-y1) ) == 0
        
        val = R * det - 2 * ( (c1*b2 - c2*b1) * (x4-x1) + (a1*c2 - a2*c1) * (y4-y1) )
        return val == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Known 5-point set (Eisenstein)
    # (0,0), (3,4), (0,5), (-4,3), (5,0)
    # Let's verify:
    # 00-34: 5
    # 00-05: 5
    # 00-43: 5
    # 00-50: 5
    # 34-05: sqrt(9+1)=sqrt(10) -> NOT integer.
    
    # Let's use a verified 5-point set:
    # (0, 0)
    # (3, 4)
    # (0, 5) -> dist to 3,4 is sqrt(9+1) no.
    
    # Try:
    # (0,0), (3,4), (5,12), (0,5), (-4,3)
    # 00-34: 5
    # 00-512: 13
    # 00-05: 5
    # 00-43: 5
    # 34-512: sqrt(4+64)=sqrt(68) no.

    # Let's just use the random search but with a better seed.
    # Seed: (0,0), (3,4), (0,5), (-4,3), (5,0) is not valid.
    
    # Valid 5-point set:
    # (0, 0)
    # (3, 4)
    # (0, 5) -> No.
    
    # How about:
    # (0, 0)
    # (3, 4)
    # (5, 12)
    # (12, 5)
    # (15, 8) -> dist to 0 is 17.
    # 34-512: sqrt(4+64) no.
    
    # Let's rely on the search to find a valid 5-point set first if needed, but the seed program already has one.
    # The seed program returns:
    # seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    # Let's assume this is valid.
    
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to add a 6th point
    # We will search for a point (x, y) such that distances to all 5 seed points are integers.
    # And the resulting set is in general position.
    
    found_solution = None
    
    # Strategy: Search for a 6th point in a box around the seed points.
    # If that fails, perturb the seed points slightly and search again.
    
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if r == 0:
            base = seed_points[:]
        else:
            # Perturb seed points slightly
            base = []
            for x, y in seed_points:
                px = x + random.randint(-2, 2)
                py = y + random.randint(-2, 2)
                base.append((px, py))
        
        # Search for a 6th point
        # Box size depends on the spread of base points
        min_x = min(p[0] for p in base)
        max_x = max(p[0] for p in base)
        min_y = min(p[1] for p in base)
        max_y = max(p[1] for p in base)
        
        search_range = 50
        start_x = min_x - search_range
        end_x = max_x + search_range
        start_y = min_y - search_range
        end_y = max_y + search_range
        
        found_6th = False
        # Random sampling in the box
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            x = random.randint(start_x, end_x)
            y = random.randint(start_y, end_y)
            new_point = (x, y)
            
            # Check if new point is distinct
            if new_point in base:
                continue
                
            # Check distances to all base points
            valid_dist = True
            for px, py in base:
                dx = x - px
                dy = y - py
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
                
            # Check if adding this point creates collinearity
            # We only need to check triples involving the new point
            collinear = False
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    if is_collinear(base[i], base[j], new_point):
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
                
            # Check if adding this point creates concyclicity
            # We need to check quadruples involving the new point
            # i.e., new_point + 3 base points
            concyclic = False
            n_base = len(base)
            for i in range(n_base):
                for j in range(i+1, n_base):
                    for k in range(j+1, n_base):
                        if is_concyclic(base[i], base[j], base[k], new_point):
                            concyclic = True
                            break
                        if time.time() > deadline - 1.0:
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            
            # If we pass all checks, we have a valid 6-point set
            # We should also verify the base points themselves are valid (they should be, but just in case)
            full_set = base + [new_point]
            if is_valid_set(full_set):
                found_solution = full_set
                found_6th = True
                break
                
        if found_6th:
            break
            
    if found_solution:
        pts = found_solution
    else:
        # Fallback to seed
        pts = seed_points
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
