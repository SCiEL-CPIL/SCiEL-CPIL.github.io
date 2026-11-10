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

    # Precompute all triple indices
    triples = list(combinations(range(N), 3))
    num_triples = len(triples)

    def compute_scores(pts):
        """Compute max multiplicity of any positive doubled area."""
        # Use a dict for counting areas
        area_counts = {}
        for i, j, k in triples:
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if d > 0:
                area_counts[d] = area_counts.get(d, 0) + 1
        if not area_counts:
            return 0, 0
        max_count = max(area_counts.values())
        best_area = max(area_counts, key=area_counts.get)
        return max_count, best_area

    def try_move(pts, idx, new_pt, current_score):
        """Try moving point idx to new_pt, return new score if improved."""
        if new_pt in pts:
            return None
        old_pt = pts[idx]
        pts[idx] = new_pt
        new_score, new_area = compute_scores(pts)
        if new_score > current_score:
            return new_score, new_area
        pts[idx] = old_pt
        return None

    # Try multiple initial configurations
    initial_configs = [
        # 6x5 grid
        [(i, j) for i in range(6) for j in range(5)],
        # 5x6 grid
        [(i, j) for i in range(5) for j in range(6)],
        # Hexagonal-like pattern
        [(i*2 + (j%2), j*3) for i in range(5) for j in range(6)],
        # Sparse grid
        [(i*3, j*3) for i in range(4) for j in range(4)][:30] + [(29, 0), (29, 1)],
    ]

    best_score = 0
    best_pts = None

    for init_pts in initial_configs:
        if len(init_pts) != N:
            continue
        # Check distinctness
        if len(set(init_pts)) != N:
            continue
        # Check bounds
        if any(x < -bounds or x > bounds or y < -bounds or y > bounds for x, y in init_pts):
            continue

        pts = init_pts[:]
        current_score, current_area = compute_scores(pts)

        if current_score > best_score:
            best_score = current_score
            best_pts = pts[:]

        # Local search from this starting point
        stagnation_count = 0
        max_stagnation = 3000

        while time.time() < deadline - 5:  # Leave 5s margin
            if stagnation_count > max_stagnation:
                # Perturb: random restart from best
                pts = best_pts[:]
                for _ in range(random.randint(3, 7)):
                    idx = random.randint(0, N-1)
                    x, y = pts[idx]
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
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

            # Pick random point and try all 8 neighbors
            idx = random.randint(0, N-1)
            x, y = pts[idx]
            improved = False

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
                        continue
                    new_pt = (nx, ny)
                    result = try_move(pts, idx, new_pt, current_score)
                    if result is not None:
                        current_score, current_area = result
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_pts = pts[:]
                        break
                if improved:
                    break

            if improved:
                stagnation_count = 0
            else:
                stagnation_count += 1

    return best_pts
# EVOLVE-BLOCK-END
