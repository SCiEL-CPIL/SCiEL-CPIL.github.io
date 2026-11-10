import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    # Known valid set of 5 points
    # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    # Let's verify this set is valid.
    # Distances:
    # (0,0) to (-56,33): 56^2+33^2 = 3136+1089 = 4225 = 65^2. OK.
    # (0,0) to (-16,63): 16^2+63^2 = 256+3969 = 4225 = 65^2. OK.
    # (0,0) to (0,33): 33. OK.
    # (0,0) to (16,63): 65. OK.
    # (-56,33) to (-16,63): 40^2 + 30^2 = 1600+900=2500=50^2. OK.
    # (-56,33) to (0,33): 56. OK.
    # (-56,33) to (16,63): 72^2 + 30^2 = 5184+900=6084=78^2. OK.
    # (-16,63) to (0,33): 16^2+30^2=256+900=1156=34^2. OK.
    # (-16,63) to (16,63): 32. OK.
    # (0,33) to (16,63): 16^2+30^2=34^2. OK.
    
    # Collinearity:
    # No three seem collinear.
    # Concyclicity:
    # Need to check.
    
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of 4x4 matrix
        # | x^2+y^2 x y 1 |
        # Using cofactor expansion
        M = [
            [x1*x1 + y1*y1, x1, y1, 1],
            [x2*x2 + y2*y2, x2, y2, 1],
            [x3*x3 + y3*y3, x3, y3, 1],
            [x4*x4 + y4*y4, x4, y4, 1]
        ]
        
        a,b,c,d = M[0]
        e,f,g,h = M[1]
        i,j,k,l = M[2]
        m,n,o,p = M[3]
        
        det = (a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) -
               b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)) +
               c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m)) -
               d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m)))
               
        return det == 0

    def validate_addition(new_pt, current_pts):
        # Check distance to all existing
        for p in current_pts:
            dx = new_pt[0] - p[0]
            dy = new_pt[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
        
        # Check collinearity with all pairs
        n = len(current_pts)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(new_pt, current_pts[i], current_pts[j]):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(new_pt, current_pts[i], current_pts[j], current_pts[k]):
                        return False
                            
        return True

    def try_extend(current_pts, search_radius):
        # Search for a new point in a box of size search_radius
        # To avoid re-adding existing points, we can just check distinctness
        # We search in a box centered at (0,0) or around the current set?
        # The current set is around (-16, 33) to (16, 63).
        # Let's search in a box around the centroid.
        cx = sum(p[0] for p in current_pts) // len(current_pts)
        cy = sum(p[1] for p in current_pts) // len(current_pts)
        
        # Search in a box of half-width search_radius around centroid
        # But we also want to allow points far away.
        # Let's just search in a box [-search_radius, search_radius]
        # If the current points are large, we might miss them.
        # But we are adding *new* points.
        
        # To be safe, search in a box that covers the current points plus some margin.
        min_x = min(p[0] for p in current_pts) - search_radius
        max_x = max(p[0] for p in current_pts) + search_radius
        min_y = min(p[1] for p in current_pts) - search_radius
        max_y = max(p[1] for p in current_pts) + search_radius
        
        # Limit the box size to avoid huge searches
        # If the range is too large, reduce it.
        # For the first extension (5->6), radius 100 is fine.
        
        existing = set(current_pts)
        
        for x in range(min_x, max_x + 1):
            if time.time() - start_time > time_limit - 10:
                return None
            for y in range(min_y, max_y + 1):
                pt = (x, y)
                if pt in existing:
                    continue
                if validate_addition(pt, current_pts):
                    return pt
        return None

    # Start with base points
    current_pts = list(base_points)
    
    # Try to extend to 6, 7, 8...
    # Increase search radius gradually
    search_radius = 50
    while len(current_pts) < 12:
        if time.time() - start_time > time_limit - 20:
            break
            
        new_pt = try_extend(current_pts, search_radius)
        if new_pt:
            current_pts.append(new_pt)
            # If we found a point, maybe we can try a smaller radius for the next one?
            # Or keep increasing?
            # Let's keep increasing to allow for spread.
            search_radius += 20
        else:
            # If we didn't find a point, try a larger radius?
            # Or give up?
            # Let's try a few larger radii.
            found = False
            for r in [100, 200, 400]:
                if time.time() - start_time > time_limit - 10:
                    break
                new_pt = try_extend(current_pts, r)
                if new_pt:
                    current_pts.append(new_pt)
                    found = True
                    search_radius = r + 20
                    break
            if not found:
                break
                
    # Finalize
    if len(current_pts) >= 3:
        pts = []
        for x, y in current_pts:
            pts.append([[x, 1], [y, 1]])
        return {'D': D, 'points': pts}
    else:
        # Fallback (should not happen)
        pts = []
        for x, y in base_points:
            pts.append([[x, 1], [y, 1]])
        return {'D': D, 'points': pts}
