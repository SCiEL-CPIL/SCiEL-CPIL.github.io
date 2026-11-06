import time
import random
import math
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    def make_grid(w, h, ox=0, oy=0):
        pts = []
        for x in range(ox, ox + w):
            for y in range(oy, oy + h):
                pts.append((x, y))
        return pts

    # Generate diverse initial candidates
    candidates = []
    
    # Various grid shapes centered
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]:
        ox = -w // 2
        oy = -h // 2
        candidates.append(make_grid(w, h, ox, oy))
        
    # Some other structured patterns
    # Diamond / Cross
    for radius in [5, 6]:
        pts = [(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1) 
               if abs(x) + abs(y) <= radius]
        if len(pts) >= 30:
            # Take first 30 or a random subset
            candidates.append(pts[:30])
            
    # Random clusters
    for _ in range(10):
        cx, cy = random.randint(-20, 20), random.randint(-20, 20)
        pts = set()
        while len(pts) < 30:
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-5, 5)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        candidates.append(list(pts))
        
    # Fully random
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))

    best_points = None
    best_score = -1
    
    def evaluate(points):
        areas = []
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                for k in range(j+1, n):
                    xk, yk = points[k]
                    area = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                    if area > 0:
                        areas.append(area)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        # Fallback
        best_points = make_grid(6, 5)
        best_score = evaluate(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    alpha = 0.9995
    max_iter = 50000
    
    # Precompute indices for faster evaluation? 
    # The evaluate function is O(N^3). N=30. 30*29*28/6 = 4060 triples.
    # This is fast enough in pure Python if optimized slightly, but let's stick to the simple version first.
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb
        new_points = current_points[:]
        idx = random.randint(0, 29)
        x, y = new_points[idx]
        
        # Random move: change x, y, or both
        r = random.random()
        if r < 0.33:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = 0
        elif r < 0.66:
            dx = 0
            dy = random.choice([-3, -2, -1, 1, 2, 3])
        else:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = random.choice([-3, -2, -1, 1, 2, 3])
            
        new_x = x + dx
        new_y = y + dy
        
        # Check bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check collision
        new_pt = (new_x, new_y)
        if new_pt in current_points:
            # If it collides with the point itself (dx=0, dy=0 is possible in logic above? 
            # No, dx and dy choices don't include 0 together, but one can be 0.
            # If dx=0, dy!=0, new_pt != (x,y).
            # If dx!=0, dy=0, new_pt != (x,y).
            # If both non-zero, new_pt != (x,y).
            # So new_pt is always different from (x,y) unless dx=0 and dy=0 which is not in choices.
            # Wait, if I pick dx from [-3..3] excluding 0? No, I used specific choices.
            # dx choices: [-3, -2, -1, 1, 2, 3] -> no 0.
            # dy choices: [-3, -2, -1, 1, 2, 3] -> no 0.
            # So new_pt is never (x,y).
            continue
                
        new_points[idx] = new_pt
        
        new_score = evaluate(new_points)
        
        # Acceptance
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_points = new_points
                current_score = new_score
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    def make_grid(w, h, ox=0, oy=0):
        pts = []
        for x in range(ox, ox + w):
            for y in range(oy, oy + h):
                pts.append((x, y))
        return pts

    # Generate diverse initial candidates
    candidates = []
    
    # Various grid shapes centered
    for w, h in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]:
        ox = -w // 2
        oy = -h // 2
        candidates.append(make_grid(w, h, ox, oy))
        
    # Some other structured patterns
    for radius in [5, 6]:
        pts = [(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1) 
               if abs(x) + abs(y) <= radius]
        if len(pts) >= 30:
            candidates.append(pts[:30])
            
    # Random clusters
    for _ in range(10):
        cx, cy = random.randint(-20, 20), random.randint(-20, 20)
        pts = set()
        while len(pts) < 30:
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-5, 5)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        candidates.append(list(pts))
        
    # Fully random
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))

    best_points = None
    best_score = -1
    
    def evaluate(points):
        areas = []
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                for k in range(j+1, n):
                    xk, yk = points[k]
                    area = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                    if area > 0:
                        areas.append(area)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())

    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        best_points = make_grid(6, 5)
        best_score = evaluate(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    T = 100.0
    T_min = 0.1
    alpha = 0.9995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        new_points = current_points[:]
        idx = random.randint(0, 29)
        x, y = new_points[idx]
        
        r = random.random()
        if r < 0.33:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = 0
        elif r < 0.66:
            dx = 0
            dy = random.choice([-3, -2, -1, 1, 2, 3])
        else:
            dx = random.choice([-3, -2, -1, 1, 2, 3])
            dy = random.choice([-3, -2, -1, 1, 2, 3])
            
        new_x = x + dx
        new_y = y + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        if new_pt in current_points:
            continue
                
        new_points[idx] = new_pt
        
        new_score = evaluate(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_points = new_points
                current_score = new_score
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points
# EVOLVE-BLOCK-END
