import random
from math import gcd
import time

def count_ordinary(points):
    n = len(points)
    ordinary = 0
    lines_seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in lines_seen:
                continue
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if a*xk + b*yk == c:
                    count += 1
            lines_seen.add(key)
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    n = 12
    limit = 10**9
    
    # Heuristic Initial State: 
    # Try to create a configuration with high collinearity.
    # A 3x4 grid is a good start.
    def make_initial_grid():
        pts = []
        # 3 rows, 4 cols
        for r in range(3):
            for c in range(4):
                pts.append([c*10, r*10])
        return pts

    def make_random_cluster():
        # Cluster points around a few centers to encourage collinearity
        centers = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(4)]
        pts = []
        for i in range(n):
            c = centers[i % 4]
            x = c[0] + random.randint(-5, 5)
            y = c[1] + random.randint(-5, 5)
            pts.append([x, y])
        # Ensure distinct
        seen = set()
        new_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                new_pts.append(p)
        while len(new_pts) < n:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                new_pts.append([x, y])
        return new_pts

    best_points = None
    best_score = float('inf')
    
    # Try a few different initial strategies
    initial_candidates = [
        make_initial_grid(),
        make_random_cluster(),
        make_random_cluster(),
        make_random_cluster()
    ]
    
    # Pick the best initial candidate
    for cand in initial_candidates:
        if len(set(tuple(p) for p in cand)) < n:
            continue
        score = count_ordinary(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
            
    if best_points is None:
        # Fallback
        best_points = make_initial_grid()
        best_score = count_ordinary(best_points)

    # Simulated Annealing
    current_points = best_points
    current_score = best_score
    T = 1000.0
    cooling = 0.999
    min_T = 1.0
    max_iter = 50000 # Adjust based on time budget
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        # Pick a random point to move
        idx = random.randint(0, n-1)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Bias towards small moves initially, larger later? 
        # Or just random within a box around current position.
        # Since we want to find structures, moving to a random spot in the whole space is bad.
        # Moving to a spot that might align with others is good.
        
        if random.random() < 0.5:
            # Local move
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            # Alignment move
            j = random.randint(0, n-1)
            if j == idx:
                j = (j + 1) % n
            xj, yj = current_points[j]
            # Place new point on line through xj, yj with some random slope/offset?
            # Or just pick a random slope and place it.
            # Let's pick a random point on a line passing through xj, yj.
            # dx, dy in [-10, 10]
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            if dx == 0 and dy == 0:
                continue
            new_x = xj + dx
            new_y = yj + dy
        
        if abs(new_x) > limit or abs(new_y) > limit:
            continue
            
        # Check distinctness
        is_distinct = True
        for k in range(n):
            if k == idx:
                continue
            if current_points[k][0] == new_x and current_points[k][1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = count_ordinary(current_points)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
            
        T *= cooling
        
    return best_points
