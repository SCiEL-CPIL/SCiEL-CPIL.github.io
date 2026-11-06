import time
import math
import random
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

    # Initial configuration: 4x3 grid
    init_pts = [[x, y] for x in range(4) for y in range(3)]
    update_best(init_pts, _count_ordinary(init_pts))
    
    if best_score == 6:
        return best_points

    # Simulated Annealing with guided moves
    # Pool of candidate points:
    # 1. Small integer box
    # 2. Points that complete lines with existing pairs
    
    def get_pool(current_pts):
        pool = set()
        # Small box
        for x in range(-10, 11):
            for y in range(-10, 11):
                pool.add((x, y))
        
        # Guided points: for every pair of points, find the third point on the line
        # that is an integer point within bounds.
        n = len(current_pts)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = current_pts[i]
                p2 = current_pts[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                
                # Normalize direction
                g = _gcd(dx, dy)
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # Check for integer points on this line within a reasonable range
                # p(t) = p1 + t * (dx_n, dy_n)
                # We look for t such that p(t) is integer and in bounds.
                # Since p1 is integer, p(t) is integer for all integer t.
                # We just need to check a range of t.
                # The line extends in both directions.
                for t in range(-20, 21):
                    if t == 0 or t == 1:
                        continue # Already in the set (or very close)
                    nx = p1[0] + t * dx_n
                    ny = p1[1] + t * dy_n
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        pool.add((nx, ny))
        
        return list(pool)

    # Run Simulated Annealing
    # We will run multiple restarts
    num_restarts = 10
    max_iterations = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current state
        if r == 0:
            current = [p[:] for p in init_pts]
        else:
            # Random restart from a perturbed version of best or random grid
            if random.random() < 0.5:
                current = [p[:] for p in best_points]
                # Perturb 2-3 points
                for _ in range(random.randint(2, 3)):
                    idx = random.randint(0, 11)
                    pool = get_pool(current)
                    # Pick a random point from pool
                    if pool:
                        current[idx] = list(random.choice(pool))
            else:
                # Random grid
                current = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
                # Ensure distinctness
                while len(set(map(tuple, current))) < 12:
                    idx = random.randint(0, 11)
                    current[idx] = [random.randint(-5, 5), random.randint(-5, 5)]

        current_score = _count_ordinary(current)
        if current_score < best_score:
            update_best(current, current_score)
            if best_score == 6:
                return best_points

        # Annealing schedule
        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
        T = T_start
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Generate a move: pick a random point to move
            idx = random.randint(0, 11)
            old_pt = current[idx]
            
            # Pick a new point from the pool
            pool = get_pool(current)
            if not pool:
                break
            
            # Bias towards points that are already on lines with other points?
            # For now, just random from pool
            new_pt = list(random.choice(pool))
            
            # Ensure distinctness
            if tuple(new_pt) in [tuple(p) for p in current if p is not current[idx]]:
                continue
                
            current[idx] = new_pt
            new_score = _count_ordinary(current)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    update_best(current, current_score)
                    if best_score == 6:
                        return best_points
            else:
                current[idx] = old_pt
                
            T *= cooling_rate

    return best_points
