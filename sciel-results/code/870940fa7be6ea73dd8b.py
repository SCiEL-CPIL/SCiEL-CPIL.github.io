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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial candidate: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    # Local search
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Temperature for simulated annealing
    temp = 1.0
    cooling = 0.99
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve < max_no_improve:
        idx = random.randint(0, 15)
        old_pt = current_points[idx][:]
        
        # Random move
        # Try small steps first, then larger
        if random.random() < 0.7:
            step = random.randint(1, 5)
        else:
            step = random.randint(5, 20)
            
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
            continue
            
        delta = new_score - current_score
        
        if delta > 0:
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                # Restart cooling?
                temp = 1.0
        else:
            # Accept if random() < exp(delta / temp)
            if delta == 0:
                # Always accept neutral moves to explore
                current_score = new_score
                no_improve += 1
            else:
                # delta < 0
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_score = new_score
                    no_improve += 1
                else:
                    current_points[idx] = old_pt
                    no_improve += 1
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # If we haven't improved, try a few random restarts with different seeds
    if best_score <= 10:
        for seed in range(10):
            if time.time() > deadline:
                break
            random.seed(seed + 100)
            # Perturb the initial grid
            pts = [p[:] for p in initial_points]
            for _ in range(5):
                idx = random.randint(0, 15)
                pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
            
            # Quick local search on this perturbed start
            curr_pts = [p[:] for p in pts]
            curr_score = evaluate(curr_pts)
            if curr_score < 0:
                continue
                
            for _ in range(500):
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
