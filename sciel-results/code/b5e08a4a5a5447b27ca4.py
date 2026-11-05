import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    def is_square(n):
        if n < 0:
            return False
        r = int(n**0.5)
        while r * r > n:
            r -= 1
        while (r + 1) * (r + 1) <= n:
            r += 1
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        s1 = x1*x1 + y1*y1
        s2 = x2*x2 + y2*y2
        s3 = x3*x3 + y3*y3
        s4 = x4*x4 + y4*y4
        
        # Determinant of:
        # | s1, x1, y1, 1 |
        # | s2, x2, y2, 1 |
        # | s3, x3, y3, 1 |
        # | s4, x4, y4, 1 |
        
        d1 = x2*(y3-y4) - y2*(x3-x4) + (x3*y4 - x4*y3)
        d2 = s2*(y3-y4) - y2*(s3-s4) + (s3*y4 - s4*y3)
        d3 = s2*(x3-x4) - x2*(s3-s4) + (s3*x4 - s4*x3)
        d4 = s2*(x3*y4 - x4*y3) - x2*(s3*y4 - s4*y3) + y2*(s3*x4 - s4*x3)
        
        return s1*d1 - x1*d2 + y1*d3 - d4 == 0

    # Generate Pythagorean displacement vectors
    max_coord = 80
    vectors = []
    for dx in range(-max_coord, max_coord+1):
        for dy in range(-max_coord, max_coord+1):
            if dx == 0 and dy == 0:
                continue
            if is_square(dx*dx + dy*dy):
                vectors.append((dx, dy))
    
    # Sort by distance to prefer closer points
    vectors.sort(key=lambda v: v[0]*v[0] + v[1]*v[1])
    
    def check_point_valid(p_new, current_points):
        """Check if p_new can be added to current_points"""
        n = len(current_points)
        
        # Check all distances are rational (integer for integer points)
        for p in current_points:
            if not is_square(dist_sq(p, p_new)):
                return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                if are_collinear(current_points[i], current_points[j], p_new):
                    return False
        
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_concyclic(current_points[i], current_points[j], current_points[k], p_new):
                        return False
        
        return True

    def try_extend(current_points, max_iter=5000):
        """Try to add a new point to current_points"""
        n = len(current_points)
        point_set = set(current_points)
        
        # First try systematic: check all lattice points in bounding box
        xs = [p[0] for p in current_points]
        ys = [p[1] for p in current_points]
        x_min, x_max = min(xs) - 50, max(xs) + 50
        y_min, y_max = min(ys) - 50, max(ys) + 50
        
        # Try random vectors from existing points
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            idx = random.randint(0, n - 1)
            p_base = current_points[idx]
            dx, dy = random.choice(vectors)
            p_new = (p_base[0] + dx, p_base[1] + dy)
            
            if p_new in point_set:
                continue
            
            if check_point_valid(p_new, current_points):
                return p_new
        
        return None

    # Known valid 5-point configuration
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = known_5[:]
    best_size = 5
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Start with 3 random points
        pts = [(0, 0)]
        
        # Point 2
        while True:
            dx, dy = random.choice(vectors)
            p2 = (dx, dy)
            if p2 != (0, 0):
                break
        pts.append(p2)
        
        # Point 3
        found_p3 = False
        for _ in range(5000):
            if time.time() > deadline:
                break
            base = random.choice(pts)
            dx, dy = random.choice(vectors)
            p3 = (base[0] + dx, base[1] + dy)
            if p3 in pts:
                continue
            if not is_square(dist_sq(pts[0], p3)) or not is_square(dist_sq(pts[1], p3)):
                continue
            if are_collinear(pts[0], pts[1], p3):
                continue
            pts.append(p3)
            found_p3 = True
            break
        
        if not found_p3:
            continue
        
        # Try to extend
        current = pts[:]
        while len(current) < 12:
            if time.time() > deadline:
                break
            p_new = try_extend(current, max_iter=3000)
            if p_new is None:
                break
            current.append(p_new)
        
        if len(current) > best_size:
            best_size = len(current)
            best_points = current[:]
            if best_size >= 8:
                break
    
    if best_points is None or len(best_points) < 3:
        best_points = known_5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
