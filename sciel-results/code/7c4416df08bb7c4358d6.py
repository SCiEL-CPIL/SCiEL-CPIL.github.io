import itertools
import time
import random

def compute_areas(points):
    """
    Compute the frequency of each positive doubled area.
    points: list of (x, y)
    Returns: dict {area: count}
    """
    n = len(points)
    areas = {}
    # Precompute vectors? No, just iterate triples.
    # 30 points -> 4060 triples. Fast enough.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                # Determinant
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
    return areas

def get_max_multiplicity(points):
    areas = compute_areas(points)
    if not areas:
        return 0
    return max(areas.values())

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 50  # Leave 10s margin for 60s limit
    
    # Initial candidates
    candidates = []
    
    # 1. Seed: 6x5 grid
    seed_grid = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(seed_grid)
    
    # 2. 7x4 grid (28 pts) + 2 extra
    # Try to add points to make it denser
    grid_7x4 = [(x, y) for x in range(7) for y in range(4)]
    # Add (6, 4) and (0, 4)? 
    # Or just take first 30 of 7x5
    grid_7x5 = [(x, y) for x in range(7) for y in range(5)]
    # Remove 5 random points
    for _ in range(5):
        pts = grid_7x5[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
        
    # 3. 5x7 grid (35 pts) remove 5
    grid_5x7 = [(x, y) for x in range(5) for y in range(7)]
    for _ in range(5):
        pts = grid_5x7[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)
        
    # 4. 4x8 grid (32 pts) remove 2
    grid_4x8 = [(x, y) for x in range(4) for y in range(8)]
    for _ in range(5):
        pts = grid_4x8[:]
        random.shuffle(pts)
        pts = pts[:30]
        candidates.append(pts)

    # Evaluate candidates
    best_score = -1
    best_points = None
    
    for pts in candidates:
        # Check distinct and bounds
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        score = get_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    # Local Search
    # We will try to improve the best candidate found so far
    current_points = best_points[:]
    current_score = best_score
    
    # Define neighbors: move a point by 1 in x or y
    # Or swap two points?
    
    # Simple local search: try moving each point to a nearby location
    # Since 30 points, 30 * 8 moves = 240 evaluations per iteration.
    # Each evaluation is ~4000 determinants.
    # 240 * 4000 = 1M ops. Fast.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(30):
            if (time.time() - start_time) > time_limit:
                break
            x, y = current_points[i]
            # Try 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    # Check if (nx, ny) is already in points
                    if (nx, ny) in current_points:
                        continue
                    
                    new_points = current_points[:]
                    new_points[i] = (nx, ny)
                    
                    # Quick check: distinct
                    if len(set(new_points)) != 30:
                        continue
                        
                    new_score = get_max_multiplicity(new_points)
                    if new_score > current_score:
                        current_score = new_score
                        current_points = new_points
                        improved = True
                        # Restart loop to try other points
                        break
                if improved:
                    break
            if improved:
                break
                
    return current_points
