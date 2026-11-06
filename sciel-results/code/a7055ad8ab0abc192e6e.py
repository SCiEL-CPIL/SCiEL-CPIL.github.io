import math
import random
import time
from collections import defaultdict
from itertools import combinations

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y2 - y1 if False else y1)
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
        # count = C(k,2) where k is the number of points on the line
        # C(k,2) = k(k-1)/2 = count
        # k^2 - k - 2*count = 0
        # k = (1 + sqrt(1 + 8*count)) / 2
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

def evaluate_fast(points, line_data):
    """Evaluate using precomputed line structure for speed."""
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
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Start with 4x4 grid which has 10 four-point lines
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    def try_move(points, idx, new_pt, current_score):
        """Try moving point idx to new_pt, return (new_score, valid)."""
        old_pt = points[idx][:]
        points[idx] = new_pt
        
        # Check distinctness
        for i, p in enumerate(points):
            if i != idx and p == new_pt:
                points[idx] = old_pt
                return -1, False
        
        s = evaluate(points)
        if s < 0:
            points[idx] = old_pt
            return -1, False
        
        points[idx] = old_pt
        return s, True
    
    def local_search(points, max_iters, time_limit):
        """Enhanced local search with multiple move strategies."""
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 2.0
        cooling = 0.997
        no_improve = 0
        max_no_improve = 5000
        iter_count = 0
        
        while iter_count < max_iters:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Multiple move strategies
            strategy = random.random()
            
            if strategy < 0.35:
                # Small perturbation
                step = random.randint(1, 3)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            elif strategy < 0.55:
                # Medium perturbation
                step = random.randint(3, 8)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            elif strategy < 0.70:
                # Large jump
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif strategy < 0.85:
                # Swap-like: move towards another point
                other_idx = random.randint(0, 15)
                if other_idx != idx:
                    other = current_points[other_idx]
                    t = random.randint(1, 10) / 10.0
                    new_x = int(old_pt[0] + t * (other[0] - old_pt[0]))
                    new_y = int(old_pt[1] + t * (other[1] - old_pt[1]))
                else:
                    new_x = old_pt[0] + random.randint(-5, 5)
                    new_y = old_pt[1] + random.randint(-5, 5)
            else:
                # Grid-aligned move: snap to nearby grid
                gx = round(old_pt[0] / 3) * 3 + random.randint(-2, 2)
                gy = round(old_pt[1] / 3) * 3 + random.randint(-2, 2)
                new_x = gx
                new_y = gy
            
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
            
            new_pt = [new_x, new_y]
            if new_pt == old_pt:
                continue
            
            # Apply move
            current_points[idx] = new_pt
            
            # Check distinctness
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                current_points[idx] = old_pt
                no_improve += 1
                continue
            
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
                    temp = 2.0
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
            if temp < 0.01:
                temp = 0.01
            
            if no_improve > max_no_improve:
                break
        
        return best_local, best_local_score
    
    # Phase 1: Optimize the initial grid
    pts, score = local_search(initial_points, max_iters=3000, time_limit=deadline - 80.0)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
    
    if best_score >= 15:
        return best_points
    
    # Phase 2: Try structured configurations
    # Attempt 1: Near-grid with systematic perturbations
    for offset in range(-2, 3):
        if time.time() > deadline - 10.0:
            break
        pts = []
        for x in range(4):
            for y in range(4):
                px = x + (offset if x == 3 else 0)
                py = y + (offset if y == 3 else 0)
                pts.append([px, py])
        s = evaluate(pts)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    # Phase 3: Random restarts with diverse configurations
    restart_attempts = 0
    max_restarts = 30
    
    while time.time() < deadline - 5.0 and restart_attempts < max_restarts:
        restart_attempts += 1
        
        mode = random.randint(0, 4)
        
        if mode == 0:
            # Perturbed 4x4 grid
            pts = [p[:] for p in initial_points]
            for i in range(16):
                if random.random() < 0.4:
                    pts[i] = [pts[i][0] + random.randint(-4, 4), pts[i][1] + random.randint(-4, 4)]
        elif mode == 1:
            # Random in small box
            pts = []
            while len(pts) < 16:
                x = random.randint(-8, 8)
                y = random.randint(-8, 8)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 2:
            # Random in medium box
            pts = []
            while len(pts) < 16:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if [x, y] not in pts:
                    pts.append([x, y])
        elif mode == 3:
            # Structured: two overlapping grids
            pts = []
            base = random.randint(-5, 5)
            for x in range(4):
                for y in range(4):
                    if x < 2 and y < 2:
                        pts.append([base + x, base + y])
                    elif x < 2 and y >= 2:
                        pts.append([base + x + 1, base + y])
                    elif x >= 2 and y < 2:
                        pts.append([base + x, base + y + 1])
                    else:
                        pts.append([base + x + 1, base + y + 1])
        else:
            # Grid with one row/column shifted
            pts = []
            shift = random.randint(1, 4)
            for x in range(4):
                for y in range(4):
                    if x == 2:
                        pts.append([x + shift, y])
                    else:
                        pts.append([x, y])
        
        s = evaluate(pts)
        if s < 0:
            continue
        
        # Run local search
        remaining = deadline - 5.0 - time.time()
        iters = max(500, min(int(remaining * 30), 2000))
        
        pts, score = local_search(pts, max_iters=iters, time_limit=deadline - 5.0)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
        if best_score >= 15:
            break
    
    # Phase 4: Final polish - intensive local search on best found
    if time.time() < deadline - 3.0:
        pts, score = local_search(best_points, max_iters=5000, time_limit=deadline - 2.0)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    return best_points
