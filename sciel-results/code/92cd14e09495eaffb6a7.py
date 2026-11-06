import time
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, _gcd(dx, c))
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = float('inf')
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]

    # The "complete quadrilateral" configuration consists of 6 points:
    # 4 vertices of a quadrilateral and the 3 intersections of its diagonals and opposite sides.
    # This configuration has 0 ordinary lines (every line contains at least 3 points? No, 
    # actually the complete quadrilateral has 4 lines with 3 points each, and 3 lines with 2 points each?
    # Let's check: 
    # Points: A, B, C, D (vertices). 
    # Lines: AB, BC, CD, DA (sides). 
    # Diagonals: AC, BD. 
    # Intersections of opposite sides: E = AB ∩ CD, F = AD ∩ BC.
    # Intersections of diagonals: G = AC ∩ BD.
    # Total 7 points. 
    # Wait, the task asks for 12 points.
    # The "Orchard-planting problem" or "Ordinary lines" minimum for n=12 is 6.
    # A known configuration achieving 6 ordinary lines for n=12 is the 4x3 grid?
    # 4x3 grid: 
    # Vertical lines: 4 lines, each with 3 points. (Non-ordinary)
    # Horizontal lines: 3 lines, each with 4 points. (Non-ordinary)
    # Diagonals? 
    # Let's count ordinary lines in 4x3 grid.
    # Total pairs: 12*11/2 = 66.
    # Lines with >=3 points:
    # 4 vertical (3 pts each) -> 4 * C(3,2) = 4*3 = 12 pairs covered.
    # 3 horizontal (4 pts each) -> 3 * C(4,2) = 3*6 = 18 pairs covered.
    # Are there other lines with >=3 points?
    # In a 4x3 grid, slopes:
    # (1,1), (1,-1), (2,1), (2,-1), (1,2), (1,-2) etc.
    # Let's just compute it.
    
    # Strategy:
    # 1. Try the 4x3 grid and its affine transforms.
    # 2. Try the "double triangle" or other symmetric configurations.
    # 3. Use a smarter local search starting from the best grid configuration.
    
    candidates = []
    
    # 1. 4x3 grid
    for x in range(4):
        for y in range(3):
            candidates.append([x, y])
    if len(candidates) == 12:
        update_best(candidates, _count_ordinary(candidates))
    
    # 2. Sheared 4x3 grids
    for k in range(-10, 11):
        if time.time() > deadline:
            break
        pts = [[x + k * y, y] for x in range(4) for y in range(3)]
        if len(set(map(tuple, pts))) != 12:
            continue
        s = _count_ordinary(pts)
        update_best(pts, s)
        if best_score == 6:
            return best_points
            
    # 3. 3x4 grid
    candidates3x4 = []
    for x in range(3):
        for y in range(4):
            candidates3x4.append([x, y])
    update_best(candidates3x4, _count_ordinary(candidates3x4))
    
    # 4. Sheared 3x4 grids
    for k in range(-10, 11):
        if time.time() > deadline:
            break
        pts = [[x + k * y, y] for x in range(3) for y in range(4)]
        if len(set(map(tuple, pts))) != 12:
            continue
        s = _count_ordinary(pts)
        update_best(pts, s)
        if best_score == 6:
            return best_points
            
    # 5. 2x6 grid
    candidates2x6 = []
    for x in range(2):
        for y in range(6):
            candidates2x6.append([x, y])
    update_best(candidates2x6, _count_ordinary(candidates2x6))
            
    # 6. Affine transforms of 4x3 grid
    base43 = [[x, y] for x in range(4) for y in range(3)]
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    if time.time() > deadline:
                        break
                    det = a * d - b * c
                    if det == 0:
                        continue
                    pts = [[a * x + b * y, c * x + d * y] for x, y in base43]
                    if len(set(map(tuple, pts))) != 12:
                        continue
                    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in pts):
                        continue
                    s = _count_ordinary(pts)
                    update_best(pts, s)
                    if best_score == 6:
                        return best_points

    # 7. Local search from the best configuration found so far
    if best_points is not None and best_score > 6:
        current = [p[:] for p in best_points]
        current_score = best_score
        
        # Pool of candidate points for replacement
        # We expand the pool slightly to allow for more complex structures
        pool = set()
        for x in range(-6, 7):
            for y in range(-6, 7):
                pool.add((x, y))
        pool = list(pool)
        
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() > deadline:
                break
                
            # Try replacing each point
            for i in range(12):
                if time.time() > deadline:
                    break
                for px, py in pool:
                    if (px, py) == tuple(current[i]):
                        continue
                    if any(tuple(current[j]) == (px, py) for j in range(12)):
                        continue
                    
                    old_pt = current[i]
                    current[i] = [px, py]
                    
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        update_best(current, new_score)
                        if best_score == 6:
                            return best_points
                        break
                    else:
                        current[i] = old_pt
                        
        # Pairwise swaps
        if best_score > 6 and time.time() < deadline:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    for px, py in pool:
                        if (px, py) == tuple(current[i]) or (px, py) == tuple(current[j]):
                            continue
                        if any(tuple(current[k]) == (px, py) for k in range(12)):
                            continue
                        
                        old_i = current[i]
                        old_j = current[j]
                        current[i] = [px, py]
                        
                        best_swap_score = current_score
                        best_swap_pt = None
                        for qx, qy in pool:
                            if (qx, qy) == tuple(old_i) or (qx, qy) == (px, py):
                                continue
                            if any(tuple(current[k]) == (qx, qy) for k in range(12)):
                                continue
                            current[j] = [qx, qy]
                            s = _count_ordinary(current)
                            if s < best_swap_score:
                                best_swap_score = s
                                best_swap_pt = [qx, qy]
                        
                        if best_swap_pt is not None:
                            current[j] = best_swap_pt
                            current_score = best_swap_score
                            improved = True
                            update_best(current, current_score)
                            if best_score == 6:
                                return best_points
                        else:
                            current[i] = old_i
                            current[j] = old_j
                            
    return best_points
