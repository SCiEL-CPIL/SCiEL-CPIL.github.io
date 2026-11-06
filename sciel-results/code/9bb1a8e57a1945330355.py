import math
import random
import time
from collections import defaultdict

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = math.gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = math.gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def get_line_points(p1, p2, bounds=1000):
    """
    Given two integer points, return the list of integer points on the line
    between them (inclusive) and potentially beyond within bounds?
    Actually, we just need to find candidate points to move TO.
    If we have a line defined by p1, p2, the primitive step is (dx/g, dy/g).
    We can generate points p1 + k * step.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return [p1]
    step_x = dx // g
    step_y = dy // g
    
    points = []
    # Search backwards from p1
    k = 0
    while True:
        px = x1 - k * step_x
        py = y1 - k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        points.append((px, py))
        k += 1
        if k > 2000: break # Safety
        
    # Search forwards from p1
    k = 1
    while True:
        px = x1 + k * step_x
        py = y1 + k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        points.append((px, py))
        k += 1
        if k > 2000: break
        
    return points

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key] += 1
    
    score = 0
    for key, count in line_counts.items():
        disc = 1 + 8 * count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc != disc:
            return -1
        k = (1 + sqrt_disc) // 2
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial candidate: 4x4 grid
    # Shift to center? Or keep 0..3?
    # Centering might help with symmetry breaking.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    def get_line_counts(points):
        line_counts = defaultdict(list)
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_hash(points[i], points[j])
                line_counts[key].append((i, j))
        # Count points per line
        point_line_counts = defaultdict(int)
        for key, pairs in line_counts.items():
            pts_on_line = set()
            for i, j in pairs:
                pts_on_line.add(i)
                pts_on_line.add(j)
            point_line_counts[key] = len(pts_on_line)
        return point_line_counts, line_counts

    def local_search(points, max_iters=5000):
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.0
        cooling = 0.999
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Strategy 1: Targeted Move (Complete a line)
            # Find lines with 3 points
            # This is expensive to do from scratch every iter.
            # Let's approximate or just do random moves with high acceptance rate for improvements.
            
            # Let's stick to random moves but with a bias towards "interesting" moves?
            # No, let's try:
            # 1. Pick a random point i.
            # 2. Pick a random point j.
            # 3. Try to move point i to a position that would make i, j, k collinear for some k?
            
            # Simpler: Just do random moves, but check if the new position creates a 4-line.
            
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Random move
            # Bias towards small moves
            if random.random() < 0.8:
                step = random.randint(1, 3)
            else:
                step = random.randint(3, 10)
                
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
                
            new_pt = [new_x, new_y]
            
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            current_points[idx] = new_pt
            new_score = evaluate(current_points)
            
            if new_score == -1:
                current_points[idx] = old_pt
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                no_improve = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
            else:
                if delta == 0:
                    current_score = new_score
                    no_improve += 1
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_points[idx] = old_pt
                    no_improve += 1
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
                
            if no_improve > 5000:
                break
                
        return best_local, best_local_score

    # Run local search
    pts, score = local_search(initial_points, max_iters=20000)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
        
    # Try a few random restarts with perturbed grids
    for _ in range(5):
        if time.time() > deadline:
            break
        pts = [p[:] for p in initial_points]
        for i in range(16):
            if random.random() < 0.3:
                pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
        
        s = evaluate(pts)
        if s < 0:
            continue
            
        pts, score = local_search(pts, max_iters=10000)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]

    return best_points
