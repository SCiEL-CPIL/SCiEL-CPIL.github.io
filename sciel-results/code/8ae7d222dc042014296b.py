import time
import random
from itertools import combinations
from collections import Counter

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    N = 30
    bounds = 30
    
    def get_points():
        # Start with a grid to have a good initial structure
        # 6x5 grid
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append((i, j))
        return pts

    def compute_scores(pts):
        # pts is a list of (x, y)
        # Compute doubled areas for all triples
        areas = []
        # Optimization: precompute? No, N is small.
        # Use list comprehension or loop
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i+1, N):
                x2, y2 = pts[j]
                for k in range(j+1, N):
                    x3, y3 = pts[k]
                    # Doubled area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if d > 0:
                        areas.append(d)
        
        if not areas:
            return 0, 0
            
        counts = Counter(areas)
        max_count = max(counts.values())
        best_area = counts.most_common(1)[0][0]
        return max_count, best_area

    # Initial points
    pts = get_points()
    # Randomize slightly to break symmetry? Or keep grid?
    # Grid is good for small areas.
    # Let's shuffle indices to make local search random? No, positions matter.
    
    current_score, current_area = compute_scores(pts)
    best_score = current_score
    best_pts = pts[:]
    
    # Local Search: Hill Climbing
    # Moves: change one point's x or y by +/- 1, or both.
    
    stagnation_count = 0
    max_stagnation = 5000 # If no improvement for 5000 steps, restart or perturb
    
    while time.time() < deadline:
        # Check time every few steps
        if stagnation_count > max_stagnation:
            # Perturb best points
            pts = best_pts[:]
            for _ in range(5):
                idx = random.randint(0, N-1)
                x, y = pts[idx]
                # Random jump
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                nx, ny = x + dx, y + dy
                if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                    # Check distinctness
                    new_pt = (nx, ny)
                    if new_pt not in pts:
                        pts[idx] = new_pt
            current_score, current_area = compute_scores(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts[:]
            stagnation_count = 0
            continue

        # Pick random point and move
        idx = random.randint(0, N-1)
        x, y = pts[idx]
        
        # Random direction
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            continue
            
        nx, ny = x + dx, y + dy
        
        # Bounds check
        if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
            stagnation_count += 1
            continue
            
        # Distinctness check
        new_pt = (nx, ny)
        if new_pt in pts:
            stagnation_count += 1
            continue
            
        # Apply move
        old_pt = pts[idx]
        pts[idx] = new_pt
        
        # Evaluate
        new_score, new_area = compute_scores(pts)
        
        if new_score > current_score:
            current_score = new_score
            current_area = new_area
            stagnation_count = 0
            if current_score > best_score:
                best_score = current_score
                best_pts = pts[:]
        else:
            # Revert
            pts[idx] = old_pt
            stagnation_count += 1
            
    return best_pts

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    from collections import Counter

    start_time = time.time()
    deadline = start_time + 50.0
    N = 30
    bounds = 30

    def compute_scores(pts):
        areas = []
        for i in range(N):
            x1, y1 = pts[i]
            for j in range(i+1, N):
                x2, y2 = pts[j]
                for k in range(j+1, N):
                    x3, y3 = pts[k]
                    d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0, 0
        counts = Counter(areas)
        max_count = max(counts.values())
        best_area = counts.most_common(1)[0][0]
        return max_count, best_area

    # Initial grid
    pts = [(i, j) for i in range(6) for j in range(5)]
    current_score, current_area = compute_scores(pts)
    best_score = current_score
    best_pts = pts[:]

    stagnation_count = 0
    max_stagnation = 2000

    while time.time() < deadline:
        if stagnation_count > max_stagnation:
            # Perturb
            pts = best_pts[:]
            for _ in range(5):
                idx = random.randint(0, N-1)
                x, y = pts[idx]
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                nx, ny = x + dx, y + dy
                if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                    new_pt = (nx, ny)
                    if new_pt not in pts:
                        pts[idx] = new_pt
            current_score, current_area = compute_scores(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts[:]
            stagnation_count = 0
            continue

        idx = random.randint(0, N-1)
        x, y = pts[idx]
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            continue
            
        nx, ny = x + dx, y + dy
        if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
            stagnation_count += 1
            continue
            
        new_pt = (nx, ny)
        if new_pt in pts:
            stagnation_count += 1
            continue
            
        old_pt = pts[idx]
        pts[idx] = new_pt
        new_score, new_area = compute_scores(pts)
        
        if new_score > current_score:
            current_score = new_score
            current_area = new_area
            stagnation_count = 0
            if current_score > best_score:
                best_score = current_score
                best_pts = pts[:]
        else:
            pts[idx] = old_pt
            stagnation_count += 1

    return best_pts
# EVOLVE-BLOCK-END
