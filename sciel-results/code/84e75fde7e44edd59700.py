import math
import random
import time

def line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(dx, dy)
    if g != 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = -dy
    B = dx
    C = dy * x1 - dx * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def get_lines(points):
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_key(points[i], points[j])
            if key is not None:
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
    return lines

def score(points):
    lines = get_lines(points)
    return sum(1 for pts in lines.values() if len(pts) == 4)

def is_valid(points):
    lines = get_lines(points)
    for pts in lines.values():
        if len(pts) >= 5:
            return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(tuple(p))
        if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = -1
    
    def sa_optimize(initial_points, time_limit):
        points = [p[:] for p in initial_points]
        
        # Repair phase
        while not is_valid(points) and time.time() < time_limit:
            i = random.randint(0, 15)
            x = random.randint(-1000, 1000)
            y = random.randint(-1000, 1000)
            points[i] = [x, y]
        
        if not is_valid(points):
            return None, -1
            
        current_score = score(points)
        best_local_points = [p[:] for p in points]
        best_local_score = current_score
        
        T = 10.0
        cooling = 0.999
        
        while time.time() < time_limit:
            # Generate candidate move
            # 50% chance: Random move
            # 50% chance: Move to a point on an existing 3-line
            if random.random() < 0.5:
                i = random.randint(0, 15)
                x = random.randint(-1000, 1000)
                y = random.randint(-1000, 1000)
            else:
                lines = get_lines(points)
                three_lines = [pts for pts in lines.values() if len(pts) == 3]
                if not three_lines:
                    i = random.randint(0, 15)
                    x = random.randint(-1000, 1000)
                    y = random.randint(-1000, 1000)
                else:
                    target_pts = random.choice(three_lines)
                    # Pick one of the points on this line to move
                    i = random.choice(list(target_pts))
                    # Pick a new position on this line
                    # Line defined by two points in target_pts
                    p1_idx, p2_idx = list(target_pts)[0], list(target_pts)[1]
                    p1 = points[p1_idx]
                    p2 = points[p2_idx]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Extend line
                    # x = p1[0] + t * dx, y = p1[1] + t * dy
                    # Choose t randomly
                    t = random.randint(-5, 5)
                    if t == 0:
                        t = 1
                    x = p1[0] + t * dx
                    y = p1[1] + t * dy
            
            # Validate move
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000):
                continue
            if points[i] == [x, y]:
                continue
            if any(p == [x, y] for p in points):
                continue
                
            old_p = points[i][:]
            points[i] = [x, y]
            
            if not is_valid(points):
                points[i] = old_p
                continue
                
            new_score = score(points)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_points = [p[:] for p in points]
            else:
                points[i] = old_p
                
            T *= cooling
            if T < 0.01:
                T = 10.0 # Restart temperature occasionally? Or just break.
                # Let's just break to save time and rely on multiple restarts
                break
                
        return best_local_points, best_local_score

    # Start 1: 4x4 Grid
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    pts, sc = sa_optimize(grid_4x4, deadline)
    if pts and sc > best_score:
        best_score = sc
        best_points = pts

    # Start 2-10: Random points
    for _ in range(10):
        rand_pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(16)]
        pts, sc = sa_optimize(rand_pts, deadline)
        if pts and sc > best_score:
            best_score = sc
            best_points = pts

    # Start 11-20: Subsets of 5x5 grid
    base_grid = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
    for _ in range(10):
        subset = random.sample(base_grid, 16)
        pts, sc = sa_optimize(subset, deadline)
        if pts and sc > best_score:
            best_score = sc
            best_points = pts
            
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
