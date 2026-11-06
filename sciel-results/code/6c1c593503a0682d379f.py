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
        # count is pairs. k points -> k(k-1)/2 pairs
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

def gf16_points():
    """
    Generate 16 points in a 4x4 grid over GF(16) using a primitive polynomial.
    We use the field GF(16) = GF(2^4) with primitive polynomial x^4 + x + 1.
    Elements are represented as 0-15.
    We create a 4x4 grid where each cell (i,j) has a unique field element.
    Then we map these to integer coordinates.
    
    Actually, a better approach: use the structure of affine plane of order 4.
    The affine plane AG(2,4) has 16 points and 20 lines of 4 points each.
    No 5 points are collinear (max line size is 4).
    We need to map this to integer coordinates.
    
    AG(2,4) points: pairs (a,b) where a,b in GF(4).
    GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
    
    Lines in AG(2,4): for each slope m in GF(4) and intercept b in GF(4),
    the line y = mx + b has 4 points. Also vertical lines x = c for c in GF(4).
    Total lines: 4 slopes * 4 intercepts + 4 vertical = 20 lines.
    
    To map to integers: we need to embed GF(4) into integers.
    One approach: represent GF(4) elements as 0, 1, 2, 3 and use a nonlinear mapping.
    But this won't preserve collinearity.
    
    Alternative: use the fact that we can place the 16 points in a 4x4 grid
    with specific perturbations to create the right collinearity structure.
    
    Let me try a different approach: start with the 4x4 grid and use local search
    with a better heuristic.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    # Simulated annealing with better move generation
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    temp = 2.0
    cooling = 0.995
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        idx = random.randint(0, 15)
        old_pt = current_points[idx][:]
        
        # Move generation: try to create or extend 4-point lines
        # With some probability, do a "line-completion" move
        if random.random() < 0.6:
            # Find a 3-point line and try to add a 4th point
            # Or find a pair and try to complete to 4
            # For simplicity, just do a random move but biased toward
            # positions that might be on existing lines
            
            # Pick two other points and try to place near the line they define
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            if i == idx or j == idx or i == j:
                i = (i + 1) % 16
                j = (j + 1) % 16
                if i == idx or j == idx or i == j:
                    i = (i + 2) % 16
                    j = (j + 2) % 16
                    if i == idx or j == idx or i == j:
                        # Fallback to random
                        new_x = old_pt[0] + random.randint(-10, 10)
                        new_y = old_pt[1] + random.randint(-10, 10)
                        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                            continue
                        new_pt = [new_x, new_y]
                    else:
                        p1 = current_points[i]
                        p2 = current_points[j]
                        # Place near the line through p1 and p2
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        if dx == 0 and dy == 0:
                            new_x = old_pt[0] + random.randint(-10, 10)
                            new_y = old_pt[1] + random.randint(-10, 10)
                        else:
                            # Parametric point on or near line
                            t = random.uniform(-0.5, 1.5)
                            new_x = int(p1[0] + t * dx) + random.randint(-3, 3)
                            new_y = int(p1[1] + t * dy) + random.randint(-3, 3)
                        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                            continue
                        new_pt = [new_x, new_y]
            else:
                p1 = current_points[i]
                p2 = current_points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    new_x = old_pt[0] + random.randint(-10, 10)
                    new_y = old_pt[1] + random.randint(-10, 10)
                else:
                    t = random.uniform(-0.5, 1.5)
                    new_x = int(p1[0] + t * dx) + random.randint(-3, 3)
                    new_y = int(p1[1] + t * dy) + random.randint(-3, 3)
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                new_pt = [new_x, new_y]
        else:
            # Random move
            step = random.randint(1, 15)
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            new_pt = [new_x, new_y]
        
        # Check distinctness
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
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                temp = 2.0
        else:
            if delta == 0:
                current_score = new_score
                no_improve += 1
            else:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_score = new_score
                    no_improve += 1
                else:
                    current_points[idx] = old_pt
                    no_improve += 1
        
        temp *= cooling
        if temp < 0.05:
            temp = 0.05

    # Multi-restart with diverse initial configurations
    for restart in range(20):
        if time.time() > deadline:
            break
        random.seed(1000 + restart)
        
        # Different initial configurations
        if restart % 4 == 0:
            # 4x4 grid
            pts = [[x, y] for x in range(4) for y in range(4)]
        elif restart % 4 == 1:
            # Perturbed grid
            pts = [[x + random.randint(-2, 2), y + random.randint(-2, 2)] 
                   for x in range(4) for y in range(4)]
        elif restart % 4 == 2:
            # Two 4x2 grids side by side
            pts = []
            for x in range(8):
                for y in range(2):
                    pts.append([x, y])
        else:
            # Random within a box
            pts = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(16)]
        
        # Ensure distinct
        seen = set()
        valid_pts = []
        for p in pts:
            pt = tuple(p)
            if pt not in seen and -1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000:
                seen.add(pt)
                valid_pts.append(p)
        while len(valid_pts) < 16:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            pt = tuple(p)
            if pt not in seen:
                seen.add(pt)
                valid_pts.append(p)
        pts = valid_pts[:16]
        
        curr_score = evaluate(pts)
        if curr_score < 0:
            continue
        
        # Quick local search
        curr_pts = [p[:] for p in pts]
        for _ in range(300):
            if time.time() > deadline:
                break
            idx = random.randint(0, 15)
            old_pt = curr_pts[idx][:]
            step = random.randint(1, 10)
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            new_pt = [new_x, new_y]
            is_distinct = True
            for i, p in enumerate(curr_pts):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            curr_pts[idx] = new_pt
            new_score = evaluate(curr_pts)
            if new_score > curr_score:
                curr_score = new_score
            else:
                curr_pts[idx] = old_pt
        
        if curr_score > best_score:
            best_score = curr_score
            best_points = [p[:] for p in curr_pts]

    return best_points
