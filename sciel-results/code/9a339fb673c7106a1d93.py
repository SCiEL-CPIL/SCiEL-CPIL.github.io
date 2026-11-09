import time
import random
import math
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains >= 5 points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check for distinct points
    if len(set(map(tuple, points))) != 16:
        return -1
        
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    
    if not valid:
        return -1
    return four_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_score = -1
    
    # Helper to generate a random valid 4-point line within bounds
    def random_4pt_line():
        # Random direction (a,b) normalized
        while True:
            a = random.randint(-10, 10)
            b = random.randint(-10, 10)
            if a == 0 and b == 0:
                continue
            d = gcd(abs(a), abs(b))
            if d > 0:
                a //= d
                b //= d
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            
            # Random offset c
            c = random.randint(-200, 200)
            
            # Find 4 points on this line
            pts = []
            # Parametrize: a*x + b*y = -c
            # We can pick 4 arbitrary x values (or y if a=0) and solve for the other
            # To ensure integer coords, we need to be careful.
            # If b != 0, y = (-c - a*x) / b. For y to be integer, (-c - a*x) must be divisible by b.
            # This is tricky with random c. 
            # Alternative: Pick 4 random points first, check if they are collinear? No, that's O(N^4).
            
            # Better: Pick a base point (x0, y0) and direction (dx, dy).
            # Points: (x0 + i*dx, y0 + i*dy) for i in {0, 1, 2, 3}
            # Ensure they are within bounds.
            # dx, dy should be coprime to avoid "skipping" integer points that might create 5-collinear with others?
            # Actually, if dx, dy are not coprime, the points are not "consecutive" integer points on the line,
            # but they are still valid integer points. However, the line key will be the same.
            # Let's just pick a random direction vector (dx, dy) and a starting point.
            
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            
            # Simplify direction to primitive? Not strictly necessary for generation,
            # but helps in diversity.
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            x0 = random.randint(-100, 100)
            y0 = random.randint(-100, 100)
            
            candidates = []
            for i in range(4):
                px = x0 + i * dx
                py = y0 + i * dy
                if abs(px) <= 1000 and abs(py) <= 1000:
                    candidates.append((px, py))
                else:
                    break
            
            if len(candidates) == 4:
                return candidates
        
    def evaluate_configuration(points):
        return fast_count(points)
    
    # Strategy: Simulated Annealing on the set of 16 points.
    # But instead of random moves, we use "line-based" moves.
    # A move consists of selecting a subset of points (e.g., 4) and replacing them
    # with points from a new random line, ensuring no 5-collinear violations are introduced
    # if possible, or accepting them with low probability if score improves.
    
    # Initial state: Random set of 16 points
    def generate_initial():
        pts = set()
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.add((x, y))
        return [list(p) for p in pts]
    
    current_points = generate_initial()
    current_score = evaluate_configuration(current_points)
    if current_score < 0:
        # If invalid, fix by removing collinear points? Or just restart.
        # Let's try to find a valid seed first.
        # Grid is a safe bet.
        current_points = [[x, y] for x in range(4) for y in range(4)]
        current_score = evaluate_configuration(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Pre-calculate line keys for current points to speed up evaluation?
    # fast_count is O(16^2) = 256 ops. Very fast.
    # 140s / 256 ops is huge. We can do millions of steps.
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Pick a random subset of points to move
        # Move 4 points at a time to a new line?
        # Or move 1 point?
        # Moving 4 points allows creating a new 4-line.
        
        idxs = random.sample(range(16), 4)
        old_positions = [current_points[i][:] for i in idxs]
        
        # Generate a new line
        new_line_pts = random_4pt_line()
        
        # Check if new points collide with existing points (not in idxs)
        # And check if new points create 5-collinear with existing points.
        
        # Tentatively apply
        for i, idx in enumerate(idxs):
            current_points[idx] = new_line_pts[i]
            
        new_score = evaluate_configuration(current_points)
        
        # Check validity
        if new_score < 0:
            # Reject invalid moves
            for i, idx in enumerate(idxs):
                current_points[idx] = old_positions[i]
            continue
            
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            for i, idx in enumerate(idxs):
                current_points[idx] = old_positions[i]
                
        T *= cooling
        steps += 1
        
        # If we have found a high score, we might want to do a more focused search
        # But for now, let SA run.
        
    # Final check
    if fast_count(best_points) < 0:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -1
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        a, b = dy, -dx
        d = gcd(abs(a), abs(b))
        if d > 0:
            a //= d
            b //= d
        if a < 0 or (a == 0 and b < 0):
            a = -a
            b = -b
        c = -(a * x1 + b * y1)
        return (a, b, c)

    def fast_count(points):
        n = len(points)
        if n != 16:
            return -1
        if len(set(map(tuple, points))) != 16:
            return -1
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                line_counts[key] = line_counts.get(key, 0) + 1
        four_count = 0
        valid = True
        for count in line_counts.values():
            if count >= 5:
                valid = False
                break
            if count == 4:
                four_count += 1
        if not valid:
            return -1
        return four_count

    def random_4pt_line():
        while True:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            x0 = random.randint(-100, 100)
            y0 = random.randint(-100, 100)
            
            candidates = []
            for i in range(4):
                px = x0 + i * dx
                py = y0 + i * dy
                if abs(px) <= 1000 and abs(py) <= 1000:
                    candidates.append((px, py))
                else:
                    break
            if len(candidates) == 4:
                return candidates

    def generate_initial():
        pts = set()
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pts.add((x, y))
        return [list(p) for p in pts]

    current_points = generate_initial()
    current_score = fast_count(current_points)
    if current_score < 0:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        current_score = fast_count(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        idxs = random.sample(range(16), 4)
        old_positions = [current_points[i][:] for i in idxs]
        
        new_line_pts = random_4pt_line()
        
        for i, idx in enumerate(idxs):
            current_points[idx] = new_line_pts[i]
            
        new_score = fast_count(current_points)
        
        if new_score < 0:
            for i, idx in enumerate(idxs):
                current_points[idx] = old_positions[i]
            continue
            
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            for i, idx in enumerate(idxs):
                current_points[idx] = old_positions[i]
                
        T *= cooling
        
    if fast_count(best_points) < 0:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
# EVOLVE-BLOCK-END
