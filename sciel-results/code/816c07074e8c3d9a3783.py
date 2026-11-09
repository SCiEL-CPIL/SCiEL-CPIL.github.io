def construct():
        import math
        import time
        
        start_time = time.time()
        deadline = start_time + 145  # Safety margin
        
        # Seed 5 points
        base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        
        def is_valid_set(points):
            n = len(points)
            # Check distances
            for i in range(n):
                for j in range(i+1, n):
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    d2 = dx*dx + dy*dy
                    r = int(math.isqrt(d2))
                    if r*r != d2:
                        return False
            
            # Check collinearity
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        # Area of triangle
                        area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                        if area2 == 0:
                            return False
            
            # Check concyclicity
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            x1, y1 = points[i]
                            x2, y2 = points[j]
                            x3, y3 = points[k]
                            x4, y4 = points[l]
                            # Determinant for circle
                            # | x^2+y^2  x  y  1 |
                            # | x1^2+y1^2 x1 y1 1 |
                            # | x2^2+y2^2 x2 y2 1 |
                            # | x3^2+y3^2 x3 y3 1 |
                            # | x4^2+y4^2 x4 y4 1 |
                            # If det == 0, concyclic.
                            
                            # Simplified check:
                            # Use the property that 4 points are concyclic iff the determinant of the matrix
                            # [x, y, x^2+y^2, 1] for the 4 points is 0.
                            
                            # Calculate determinant of 4x4 matrix
                            # M = [ [x1, y1, x1^2+y1^2, 1], ... ]
                            
                            # To avoid floating point, use integer arithmetic.
                            # We can check if the circle through i,j,k passes through l.
                            # Circle eq: x^2 + y^2 + Dx + Ey + F = 0
                            # Solve for D, E, F using i, j, k.
                            # Then check if l satisfies it.
                            
                            # System:
                            # x1^2+y1^2 + Dx1 + Ey1 + F = 0
                            # x2^2+y2^2 + Dx2 + Ey2 + F = 0
                            # x3^2+y3^2 + Dx3 + Ey3 + F = 0
                            
                            # Let A1 = x1^2+y1^2, etc.
                            # D x1 + E y1 + F = -A1
                            # D x2 + E y2 + F = -A2
                            # D x3 + E y3 + F = -A3
                            
                            # Subtract 1st from 2nd and 3rd:
                            # D(x2-x1) + E(y2-y1) = -(A2-A1)
                            # D(x3-x1) + E(y3-y1) = -(A3-A1)
                            
                            a1 = x2-x1; b1 = y2-y1; c1 = -( (x2*x2+y2*y2) - (x1*x1+y1*y1) )
                            a2 = x3-x1; b2 = y3-y1; c2 = -( (x3*x3+y3*y3) - (x1*x1+y1*y1) )
                            
                            det = a1*b2 - a2*b1
                            if det == 0:
                                # Collinear, already handled, but skip
                                continue
                                
                            D = (c1*b2 - c2*b1) / det
                            E = (a1*c2 - a2*c1) / det
                            
                            # D and E are rational.
                            # F = -A1 - D*x1 - E*y1
                            
                            # Check point l:
                            # x4^2+y4^2 + D*x4 + E*y4 + F == 0
                            # x4^2+y4^2 + D*x4 + E*y4 - A1 - D*x1 - E*y1 == 0
                            # (x4^2+y4^2 - A1) + D(x4-x1) + E(y4-y1) == 0
                            
                            # Multiply by det to clear denominator:
                            # det * (x4^2+y4^2 - A1) + (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1) == 0
                            
                            val = det * (x4*x4+y4*y4 - (x1*x1+y1*y1)) + (c1*b2 - c2*b1)*(x4-x1) + (a1*c2 - a2*c1)*(y4-y1)
                            
                            if val == 0:
                                return False
                                
            return True

        # Search for a 6th point
        # We'll try to add a point to the base 5.
        # To keep it fast, we search in a limited box.
        
        best_set = base_points[:]
        
        # Try to find P5
        # We can iterate over x, y in a range
        # Given the symmetry and values of base points, let's try around (0, 100) or similar.
        
        # Actually, a known 6-point set is:
        # (0,0), (15,8), (8,15), (15,-8), (8,-15), (0, 17) is invalid.
        
        # Let's just return the 5 points if we don't find a 6th quickly.
        # But 5 is score 1.0. We want > 1.0.
        
        # I will hardcode a known valid 6-point set if I can recall one.
        # If not, I'll use the search.
        
        # Search range: -100 to 100
        found_6 = False
        for x in range(-100, 101):
            if time.time() > deadline:
                break
            for y in range(-100, 101):
                if time.time() > deadline:
                    break
                new_point = (x, y)
                # Check if new point is distinct
                if new_point in best_set:
                    continue
                candidate = best_set[:] + [new_point]
                if is_valid_set(candidate):
                    best_set = candidate
                    found_6 = True
                    break
            if found_6:
                break
        
        if found_6:
            # Try to find a 7th?
            # Might be too slow.
            pass
            
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
