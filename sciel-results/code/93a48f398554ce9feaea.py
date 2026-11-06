import time
import random
from collections import defaultdict
from math import gcd

def count_ordinary_lines(points):
    """Count ordinary lines (lines with exactly 2 points) for n=12 points."""
    n = len(points)
    if n < 2:
        return 0
    
    # For each pair of points, compute the line in normalized form
    # Line: ax + by = c where gcd(a,b,c)=1, and sign normalized
    line_count = defaultdict(int)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normal vector
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            # Normalize
            g = gcd(a, b)
            if g > 0:
                a //= g
                b //= g
                c //= g
            elif g < 0:
                a //= g
                b //= g
                c //= g
            
            # Ensure consistent sign: a > 0, or a == 0 and b > 0
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            line_count[(a, b, c)] += 1
    
    # Count lines with exactly 2 points
    ordinary = 0
    for key, cnt in line_count.items():
        if cnt == 1:  # Each pair on a line with exactly 2 points contributes 1
            ordinary += 1
    
    return ordinary

def fast_count(points):
    """Faster counting using the fact that for n=12, we can use the dual approach.
    For each point, count how many lines through it have exactly 2 points total."""
    n = len(points)
    if n < 2:
        return 0
    
    # For each pair, compute line key
    # Use a more efficient grouping
    line_points = defaultdict(list)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            a = dy
            b = -dx
            c = -(a * x1 + b * y1)
            
            g = gcd(a, b)
            if g > 0:
                a //= g
                b //= g
                c //= g
            elif g < 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            line_points[(a, b, c)].append(i)
    
    # Count ordinary lines
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Known good configurations to try
    # Start with various structured configurations
    
    configs = []
    
    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 12 points on a circle (regular 12-gon approximated with integers)
    # Use points that form a near-regular polygon
    circle_pts = []
    for i in range(12):
        import math
        angle = 2 * math.pi * i / 12
        x = round(10 * math.cos(angle))
        y = round(10 * math.sin(angle))
        circle_pts.append([x, y])
    # Remove duplicates
    seen = set()
    unique_circle = []
    for p in circle_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_circle.append(p)
    if len(unique_circle) >= 12:
        configs.append(unique_circle[:12])
    
    # Pappus configuration (9 points on 9 lines, each line has 3 points)
    # Pappus: two lines L1, L2 with 3 points each
    # L1: A1, A2, A3; L2: B1, B2, B3
    # Connecting lines: A1B1, A2B2, A3B3 intersect at C1, C2, C3
    # Additional lines: A1B2, A2B1 intersect at C3' etc.
    # Standard Pappus: 9 points, 9 lines, each line has 3 points
    # For 12 points, we can extend this
    
    # Let's use a known configuration: 3 points on each of 4 lines
    # forming a complete quadrilateral
    
    # Try: 3x4 grid variant with some points moved
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    
    # Move one point to create more 3-point lines
    # In a 3x4 grid, we have:
    # - 3 horizontal lines with 4 points each
    # - 4 vertical lines with 3 points each
    # - Various diagonal lines
    # Let's count: 3 lines with 4 pts, 4 lines with 3 pts
    # Diagonals: slope 1: (0,0)-(1,1)-(2,2), (1,0)-(2,1)-(3,2) -> 2 lines with 3 pts
    #           slope -1: (0,2)-(1,1)-(2,0), (1,2)-(2,1)-(3,0) -> 2 lines with 3 pts
    #           slope 2: (0,0)-(1,2) -> 2 pts, (0,1)-(1,3) out of range
    #           slope 1/2: (0,0)-(2,1) -> 2 pts
    # So we have 3+4+2+2 = 11 non-ordinary lines
    # Total lines = C(12,2) - (lines with 3+ pts)
    # Actually let's just compute
    
    configs.append(grid_3x4)
    
    # Try a configuration based on the Hessian configuration
    # 9 points, 12 lines, each line has 3 points
    # For 12 points, add 3 more
    
    # Use a known minimum configuration search
    # Start with the best config and do local search
    
    best_pts = None
    best_score = float('inf')
    
    for config in configs:
        if len(config) != 12:
            continue
        # Check all distinct
        seen = set()
        valid = True
        for p in config:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        if not valid:
            continue
        # Check not all collinear
        if len(config) >= 3:
            x1, y1 = config[0]
            x2, y2 = config[1]
            x3, y3 = config[2]
            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                # Check if all are collinear
                all_collinear = True
                for p in config[3:]:
                    if (p[0] - x1) * (y3 - y1) != (p[1] - y1) * (x3 - x1):
                        all_collinear = False
                        break
                if all_collinear:
                    continue
        
        score = fast_count(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        # Fallback: random points
        random.seed(42)
        best_pts = []
        seen = set()
        while len(best_pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in seen:
                seen.add((x, y))
                best_pts.append([x, y])
        best_score = fast_count(best_pts)
    
    current_pts = [list(p) for p in best_pts]
    current_score = best_score
    
    # Local search with simulated annealing
    temp = 5.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Generate candidate positions
        candidates = []
        
        # Try to place on existing lines with 3+ points
        # First, find all lines among the other 11 points
        line_points = defaultdict(list)
        for i in range(12):
            for j in range(i + 1, 12):
                if i == idx or j == idx:
                    continue
                x1, y1 = current_pts[i]
                x2, y2 = current_pts[j]
                dx = x2 - x1
                dy = y2 - y1
                a = dy
                b = -dx
                c = -(a * x1 + b * y1)
                g = gcd(a, b)
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                elif g < 0:
                    a //= g
                    b //= g
                    c //= g
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                line_points[(a, b, c)].append(i)
        
        # For lines with 3+ points, try to place idx on them
        for line_key, other_indices in line_points.items():
            if len(other_indices) >= 3:
                a, b, c = line_key
                # Find integer points on this line
                # ax + by = c
                if b != 0:
                    # y = (c - ax) / b
                    # Try various x values
                    for x in range(-30, 31):
                        val = c - a * x
                        if val % b == 0:
                            y = val // b
                            new_pt = (x, y)
                            if new_pt == (old_x, old_y):
                                continue
                            if any(new_pt == tuple(p) for p in current_pts):
                                continue
                            candidates.append(new_pt)
                elif a != 0:
                    # x = c/a
                    if c % a == 0:
                        x = c // a
                        for y in range(-30, 31):
                            new_pt = (x, y)
                            if new_pt == (old_x, old_y):
                                continue
                            if any(new_pt == tuple(p) for p in current_pts):
                                continue
                            candidates.append(new_pt)
        
        # Small perturbations
        for _ in range(5):
            nx = old_x + random.randint(-3, 3)
            ny = old_y + random.randint(-3, 3)
            candidates.append((nx, ny))
        
        # Larger jumps
        for _ in range(3):
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            candidates.append((nx, ny))
        
        # Try candidates
        moved = False
        random.shuffle(candidates)
        
        for nx, ny in candidates:
            if (nx, ny) == (old_x, old_y):
                continue
            if any((nx, ny) == tuple(p) for p in current_pts):
                continue
            
            current_pts[idx] = [nx, ny]
            new_score = fast_count(current_pts)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [list(p) for p in current_pts]
                moved = True
                break
            else:
                delta = new_score - current_score
                if delta == 0 or random.random() < pow(2.718, -delta / temp):
                    current_score = new_score
                    moved = True
                    break
                else:
                    current_pts[idx] = [old_x, old_y]
        
        if not moved:
            current_pts[idx] = [old_x, old_y]
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
        
        # Early exit if we reached the theoretical minimum
        if best_score <= 6:
            break
    
    return best_pts
