import math
import random
import time
from fractions import Fraction

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            if a == 0 and b == 0:
                continue
                
            g = math.gcd(a, math.gcd(b, c))
            if g < 0:
                g = -g
            if g != 0:
                a //= g
                b //= g
                c //= g
                
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
            
    return ordinary

def _line_through_two_points(p1, p2):
    """Return (a, b, c) normalized for line ax + by + c = 0 through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    if a == 0 and b == 0:
        return None
        
    g = math.gcd(a, math.gcd(b, c))
    if g < 0:
        g = -g
    if g != 0:
        a //= g
        b //= g
        c //= g
        
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def _find_integer_points_on_line(line_key, existing_points, max_coord=10**9):
    """Find integer points on the given line that are not in existing_points."""
    a, b, c = line_key
    points = []
    
    if a == 0 and b == 0:
        return points
        
    # Line: ax + by + c = 0
    # Find integer solutions
    
    if a == 0:
        # by + c = 0 -> y = -c/b (if b divides c)
        if b != 0 and c % b == 0:
            y = -c // b
            for x in range(-max_coord, max_coord + 1):
                p = (x, y)
                if p not in existing_points:
                    points.append(p)
        return points
        
    if b == 0:
        # ax + c = 0 -> x = -c/a (if a divides c)
        if a != 0 and c % a == 0:
            x = -c // a
            for y in range(-max_coord, max_coord + 1):
                p = (x, y)
                if p not in existing_points:
                    points.append(p)
        return points
        
    # General case: ax + by = -c
    # Find gcd(a, b)
    g = math.gcd(a, b)
    if c % g != 0:
        return points  # No integer solutions
        
    a_prime = a // g
    b_prime = b // g
    c_prime = c // g
    
    # Find one particular solution to a'x + b'y = -c'
    # Use extended Euclidean algorithm
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = extended_gcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
        
    g_ab, x0, y0 = extended_gcd(a_prime, b_prime)
    # a_prime * x0 + b_prime * y0 = g_ab = 1
    # So a_prime * (x0 * (-c_prime)) + b_prime * (y0 * (-c_prime)) = -c_prime
    x_particular = x0 * (-c_prime)
    y_particular = y0 * (-c_prime)
    
    # General solution: x = x_particular + b_prime * t, y = y_particular - a_prime * t
    # Find t range so that |x|, |y| <= max_coord
    
    points_found = []
    # t range: |x_particular + b_prime * t| <= max_coord
    # and |y_particular - a_prime * t| <= max_coord
    
    t_min = -max_coord
    t_max = max_coord
    
    for t in range(t_min, t_max + 1):
        x = x_particular + b_prime * t
        y = y_particular - a_prime * t
        
        if abs(x) > max_coord or abs(y) > max_coord:
            continue
            
        p = (x, y)
        if p not in existing_points:
            points_found.append(p)
            
    return points_found

def _generate_line_intersection_config(num_lines, seed=42):
    """Generate points as intersections of lines."""
    random.seed(seed)
    
    # Choose lines with small integer coefficients
    lines = []
    for _ in range(num_lines):
        # Random line: ax + by + c = 0 with small coefficients
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        if a == 0 and b == 0:
            a = 1
        c = random.randint(-50, 50)
        
        # Normalize
        g = math.gcd(a, math.gcd(b, c))
        if g < 0:
            g = -g
        if g != 0:
            a //= g
            b //= g
            c //= g
            
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
            
        line_key = (a, b, c)
        if line_key not in lines:
            lines.append(line_key)
            
    # Find all pairwise intersections
    points = set()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, c1 = lines[i]
            a2, b2, c2 = lines[j]
            
            # Solve:
            # a1*x + b1*y = -c1
            # a2*x + b2*y = -c2
            
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue  # Parallel or same line
                
            x_num = (-c1) * b2 - (-c2) * b1
            y_num = a1 * (-c2) - a2 * (-c1)
            
            # Check if x_num and y_num are divisible by det
            if x_num % det != 0 or y_num % det != 0:
                continue
                
            x = x_num // det
            y = y_num // det
            
            if abs(x) <= 10**9 and abs(y) <= 10**9:
                points.add((x, y))
                
    return list(points)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = float('inf')
    
    # Strategy 1: Line intersection configurations
    for num_lines in range(5, 15):
        if time.time() > deadline:
            break
            
        for seed in range(20):
            if time.time() > deadline:
                break
                
            pts = _generate_line_intersection_config(num_lines, seed)
            
            # Need exactly 12 distinct points
            if len(pts) != 12:
                continue
                
            # Check not all collinear
            if len(pts) < 3:
                continue
                
            # Check if all collinear
            p0, p1, p2 = pts[0], pts[1], pts[2]
            line_key = _line_through_two_points(p0, p1)
            if line_key:
                a, b, c = line_key
                all_collinear = all(a * p[0] + b * p[1] + c == 0 for p in pts)
                if all_collinear:
                    continue
                    
            score = _count_ordinary_lines(pts)
            
            if score < best_score:
                best_score = score
                best_points = [list(p) for p in pts]
                
            if best_score == 6:
                return best_points
                
    # Strategy 2: Grid configurations
    for grid_configs in [
        [[i, j] for i in range(4) for j in range(3)],
        [[i, j] for i in range(3) for j in range(4)],
        [[i, j] for i in range(2) for j in range(6)],
        [[i, j] for i in range(6) for j in range(2)],
        [[i, j] for i in range(12) for j in range(1) if i < 12],
    ]:
        if time.time() > deadline:
            break
            
        pts = [list(p) for p in grid_configs]
        
        # Check not all collinear
        if len(pts) < 3:
            continue
            
        p0, p1, p2 = pts[0], pts[1], pts[2]
        line_key = _line_through_two_points(p0, p1)
        if line_key:
            a, b, c = line_key
            all_collinear = all(a * p[0] + b * p[1] + c == 0 for p in pts)
            if all_collinear:
                continue
                
        score = _count_ordinary_lines(pts)
        
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
        if best_score == 6:
            return best_points
            
    # Strategy 3: Known good configurations (e.g., near-regular polygons)
    # Try configurations based on roots of unity or symmetric patterns
    symmetric_configs = [
        # 3x4 grid with perturbation
        [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], [0,2], [1,2], [2,2], [3,2]],
        # Hexagonal-like
        [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], [0,2], [1,2], [2,2], [3,2]],
    ]
    
    for config in symmetric_configs:
        if time.time() > deadline:
            break
            
        pts = [list(p) for p in config]
        
        # Check not all collinear
        if len(pts) < 3:
            continue
            
        p0, p1, p2 = pts[0], pts[1], pts[2]
        line_key = _line_through_two_points(p0, p1)
        if line_key:
            a, b, c = line_key
            all_collinear = all(a * p[0] + b * p[1] + c == 0 for p in pts)
            if all_collinear:
                continue
                
        score = _count_ordinary_lines(pts)
        
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
        if best_score == 6:
            return best_points
            
    # Fallback: return best found or a valid default
    if best_points is None:
        # Default: 4x3 grid
        best_points = [[i, j] for i in range(4) for j in range(3)]
        
    return best_points
