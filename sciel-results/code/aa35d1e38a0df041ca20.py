import math
import time

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit

    # Base set of 5 points from seed
    # D = 1
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def are_concyclic(p1, p2, p3, p4):
        pts = [p1, p2, p3, p4]
        def det4(m):
            return (m[0][0]*((m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[3][2]*m[2][3] + m[2][1]*m[3][3]*m[1][2]) -
                             (m[1][1]*m[3][2]*m[2][3] + m[2][1]*m[2][3]*m[1][3] + m[1][2]*m[3][3]*m[2][2])) -
                    m[0][1]*((m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[3][2]*m[2][3] + m[2][0]*m[3][3]*m[1][2]) -
                             (m[1][0]*m[3][2]*m[2][3] + m[2][0]*m[2][3]*m[1][3] + m[1][2]*m[3][3]*m[2][0])) +
                    m[0][2]*((m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[3][1]*m[2][3] + m[2][0]*m[3][3]*m[1][1]) -
                             (m[1][0]*m[3][1]*m[2][3] + m[2][0]*m[2][3]*m[1][3] + m[1][1]*m[3][3]*m[2][0])) -
                    m[0][3]*((m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[3][1]*m[2][2] + m[2][0]*m[3][2]*m[1][1]) -
                             (m[1][0]*m[3][1]*m[2][2] + m[2][0]*m[2][2]*m[1][3] + m[1][1]*m[3][2]*m[2][0])))
            
        rows = []
        for p in pts:
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_perfect_square(dx*dx + dy*dy):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    best_points = base_points
    found_6th = False
    
    # Search for 6th integer point
    # We limit the search to a reasonable box to ensure we finish in time
    # If not found, we return the 5-point set.
    for B in range(1, 150):
        if time.time() - start_time > timeout:
            break
        for x in range(-B, B + 1):
            if time.time() - start_time > timeout:
                break
            for y in range(-B, B + 1):
                if (x, y) in base_points:
                    continue
                
                candidate = (x, y)
                
                valid_candidate = True
                for p in base_points:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_candidate = False
                        break
                
                if not valid_candidate:
                    continue
                
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        if are_collinear(base_points[i], base_points[j], candidate):
                            valid_candidate = False
                            break
                    if not valid_candidate:
                        break
                
                if not valid_candidate:
                    continue
                
                for i in range(len(base_points)):
                    for j in range(i+1, len(base_points)):
                        for k in range(j+1, len(base_points)):
                            if are_concyclic(base_points[i], base_points[j], base_points[k], candidate):
                                valid_candidate = False
                                break
                        if not valid_candidate:
                            break
                    if not valid_candidate:
                        break
                
                if valid_candidate:
                    new_set = base_points + [candidate]
                    if is_valid_set(new_set):
                        best_points = new_set
                        found_6th = True
                        break
            if found_6th:
                break

    D = 1
    points_out = []
    for p in best_points:
        x, y = p
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': points_out}
