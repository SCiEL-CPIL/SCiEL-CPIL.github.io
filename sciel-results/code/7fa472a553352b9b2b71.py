import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Helper functions for exact arithmetic checks
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_valid(points):
        n = len(points)
        if n < 3:
            return True
        
        # 1. Check distinctness
        pts_set = set(points)
        if len(pts_set) != n:
            return False
            
        # 2. Check integral distances (for D=1, squared distance must be perfect square)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    return False
        
        # 3. Check no three collinear
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    # Area of triangle = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    if area2 == 0:
                        return False
        
        # 4. Check no four concyclic
        # Equation of circle: x^2 + y^2 + Dx + Ey + F = 0
        # For 4 points, the determinant of the matrix:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        # must be non-zero.
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        # Calculate determinant
                        # Using expansion or simple formula
                        # Det = x1(y2((x3^2+y3^2) - (x4^2+y4^2)) - y2((x3^2+y3^2) - (x4^2+y4^2))) ... this is messy.
                        # Let's use the property that 4 points are concyclic iff the determinant is 0.
                        # We can compute it via row operations or just direct calculation for 4x4.
                        
                        # M = [[x, y, x^2+y^2, 1]]
                        m11, m12, m13, m14 = x1, y1, x1*x1+y1*y1, 1
                        m21, m22, m23, m24 = x2, y2, x2*x2+y2*y2, 1
                        m31, m32, m33, m34 = x3, y3, x3*x3+y3*y3, 1
                        m41, m42, m43, m44 = x4, y4, x4*x4+y4*y4, 1
                        
                        # Determinant of 4x4 matrix
                        # Using Leibniz formula or Sarrus-like expansion.
                        # For efficiency, we can subtract row 1 from rows 2,3,4 to simplify.
                        # But since n is small, let's just do the standard calculation.
                        
                        def det4(m):
                            # m is a list of 4 tuples/lists
                            a, b, c, d = m[0]
                            e, f, g, h = m[1]
                            i, j, k, l = m[2]
                            m_val, n_val, o, p = m[3]
                            
                            # Expansion along first row
                            # a * det3(b,c,d; f,g,h; j,k,l) - b * det3(e,c,d; i,g,h; m_val,k,l) + ...
                            def det3(u, v, w, x, y, z, s, t, r):
                                return u*(y*r - z*t) - v*(x*r - z*s) + w*(x*t - y*s)
                            
                            d11 = det3(f, g, h, j, k, l, n_val, o, p)
                            d12 = det3(e, g, h, i, k, l, m_val, o, p)
                            d13 = det3(e, f, h, i, j, l, m_val, n_val, p)
                            d14 = det3(e, f, g, i, j, k, m_val, n_val, o)
                            
                            return a*d11 - b*d12 + c*d13 - d*d14

                        det_val = det4([(m11, m12, m13, m14), (m21, m22, m23, m24), (m31, m32, m33, m34), (m41, m42, m43, m44)])
                        if det_val == 0:
                            return False
        
        return True

    # Initial known 5-point set
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = initial_points[:]
    
    # Search strategy:
    # 1. Try to add random points to the current best set.
    # 2. If stuck, try to modify existing points.
    
    # We will use a randomized search with a focus on adding points.
    # We'll keep the current best set and try to extend it.
    
    # To speed up, we can pre-filter candidate points.
    # A point P is a candidate if it has integral distance to all points in the current set.
    # We can generate candidates by intersecting circles.
    
    def find_candidates(current_set, limit=100):
        candidates = set()
        # Pick 2 random points from current_set to define the intersection
        # Actually, just try random points in a box around the current set.
        # Or, better: iterate over a grid of x, y and check distances.
        
        # Let's try a grid search around the current points.
        # Determine bounding box
        if not current_set:
            return []
        
        min_x = min(p[0] for p in current_set)
        max_x = max(p[0] for p in current_set)
        min_y = min(p[1] for p in current_set)
        max_y = max(p[1] for p in current_set)
        
        # Expand box
        pad = 100
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad
        
        # Random sampling in this box is faster than full grid if box is large.
        # But we want to find points that are integral to ALL points in current_set.
        # This is a strong constraint.
        
        # Alternative: Use the property that if P is integral to A and B, it lies on the intersection of two circles.
        # We can iterate over pairs in current_set, find the intersection points (if rational), and check if they are integral to the rest.
        
        # This is more efficient.
        n = len(current_set)
        for i in range(n):
            ax, ay = current_set[i]
            for j in range(i + 1, n):
                bx, by = current_set[j]
                
                # Distance between A and B
                dx = ax - bx
                dy = ay - by
                d2 = dx * dx + dy * dy
                d = math.isqrt(d2)
                if d * d != d2:
                    continue # Should not happen if current_set is valid
                
                # Intersection of circles centered at A and B with radius r1, r2?
                # We don't know r1, r2.
                # However, any point P such that dist(A,P) and dist(B,P) are integers is a candidate.
                # This is equivalent to finding integer solutions to:
                # (x-ax)^2 + (y-ay)^2 = u^2
                # (x-bx)^2 + (y-by)^2 = v^2
                # for some integers u, v.
                
                # This is still hard. Let's stick to random search but with a bias.
                # Randomly pick a point P. Check if it's integral to all points in current_set.
                # If yes, add to candidates.
        
        # Let's just do random sampling in the main loop.
        return candidates

    # Main loop
    current_points = best_points[:]
    
    # We will try to improve current_points.
    # We'll use a simple hill climbing:
    # Try to add a random point. If valid, accept.
    # If we fail to add for many steps, try to move a random point.
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Strategy 1: Add a new point
        if len(current_points) < 12:
            # Generate random point
            # Bias towards existing points to find dense clusters?
            # Or just uniform in a box.
            # Let's use a box around the centroid.
            cx = sum(p[0] for p in current_points) / len(current_points)
            cy = sum(p[1] for p in current_points) / len(current_points)
            
            # Random radius
            r = random.randint(1, 200)
            angle = random.uniform(0, 2 * math.pi)
            nx = int(cx + r * math.cos(angle))
            ny = int(cy + r * math.sin(angle))
            
            new_point = (nx, ny)
            
            # Check if new_point is distinct
            if new_point in current_points:
                continue
                
            # Check if new_point is integral to all current points
            valid_add = True
            for p in current_points:
                dx = nx - p[0]
                dy = ny - p[1]
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    valid_add = False
                    break
            
            if valid_add:
                # Check full validity (collinear, concyclic)
                if check_valid(current_points + [new_point]):
                    current_points.append(new_point)
                    if len(current_points) > len(best_points):
                        best_points = current_points[:]
        
        # Strategy 2: Move a random point
        # This helps escape local minima where we can't add points.
        if len(current_points) > 0:
            idx = random.randint(0, len(current_points) - 1)
            old_p = current_points[idx]
            
            # Generate new location
            cx = sum(p[0] for p in current_points) / len(current_points)
            cy = sum(p[1] for p in current_points) / len(current_points)
            r = random.randint(1, 200)
            angle = random.uniform(0, 2 * math.pi)
            nx = int(cx + r * math.cos(angle))
            ny = int(cy + r * math.sin(angle))
            
            new_p = (nx, ny)
            
            if new_p in current_points:
                continue
            
            # Check if new_p is integral to all OTHER points
            valid_move = True
            for k, p in enumerate(current_points):
                if k == idx:
                    continue
                dx = nx - p[0]
                dy = ny - p[1]
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    valid_move = False
                    break
            
            if valid_move:
                # Check full validity
                new_set = current_points[:]
                new_set[idx] = new_p
                if check_valid(new_set):
                    # Accept if it leads to a set that is "better" or same size
                    # For now, accept to explore.
                    current_points = new_set
                    if len(current_points) > len(best_points):
                        best_points = current_points[:]
        
        # Occasionally restart with a known small set to explore different regions
        if attempts % 10000 == 0 and time.time() < deadline - 10:
            # Restart with initial 5 points or random 3 points
            if random.random() < 0.5:
                current_points = best_points[:]
            else:
                # Random 3 points forming an integer triangle
                # Just pick random points until valid
                while True:
                    p1 = (random.randint(-100, 100), random.randint(-100, 100))
                    p2 = (random.randint(-100, 100), random.randint(-100, 100))
                    p3 = (random.randint(-100, 100), random.randint(-100, 100))
                    if p1 == p2 or p2 == p3 or p1 == p3:
                        continue
                    # Check integral distances
                    d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                    d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                    d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                    if is_square(d12) and is_square(d23) and is_square(d13):
                        # Check collinear
                        area = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
                        if area != 0:
                            current_points = [p1, p2, p3]
                            break

    # Format output
    # D = 1
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    
    result_points = []
    for x, y in best_points:
        result_points.append([ [x, 1], [y, 1] ])
        
    return {'D': 1, 'points': result_points}
