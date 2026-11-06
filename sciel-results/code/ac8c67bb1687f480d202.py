"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START
def _dist2(p1, p2):
    """Squared Euclidean distance between two points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _creates_isosceles(new_pt, existing_pts):
    """Check if adding new_pt creates any isosceles triangle with existing points."""
    n = len(existing_pts)
    for i in range(n):
        d1 = _dist2(new_pt, existing_pts[i])
        for j in range(i + 1, n):
            d2 = _dist2(new_pt, existing_pts[j])
            d3 = _dist2(existing_pts[i], existing_pts[j])
            if d1 == d2 or d1 == d3 or d2 == d3:
                return True
    return False

def _greedy_construct(seed):
    """Greedy construction: add points in order of some criterion."""
    random.seed(seed)
    points = []
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_positions)
    
    for pt in all_positions:
        if not _creates_isosceles(pt, points):
            points.append(pt)
    return points

def _local_search(points, time_budget):
    """Improve solution by swapping points."""
    start_time = time.time()
    best_points = list(points)
    best_len = len(best_points)
    
    while time.time() - start_time < time_budget:
        improved = False
        # Try removing a point and adding a different one
        for i in range(len(best_points)):
            if time.time() - start_time >= time_budget:
                break
            removed = best_points[i]
            remaining = best_points[:i] + best_points[i+1:]
            
            # Try to find a point to add
            found = False
            for x in range(N):
                if found or time.time() - start_time >= time_budget:
                    break
                for y in range(N):
                    if found or time.time() - start_time >= time_budget:
                        break
                    candidate = (x, y)
                    if candidate == removed:
                        continue
                    if not _creates_isosceles(candidate, remaining):
                        new_points = remaining + [candidate]
                        if len(new_points) > best_len:
                            best_points = new_points
                            best_len = len(new_points)
                            improved = True
                        found = True
                        break
        if not improved:
            break
    
    return best_points

def construct():
    """Construct a no-isosceles set with local search improvement."""
    start_time = time.time()
    
    # Try multiple greedy seeds and keep the best
    best_result = []
    
    for seed in range(10):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        points = _greedy_construct(seed)
        if len(points) > len(best_result):
            best_result = points
    
    # Local search on the best result
    remaining_time = TIME_LIMIT - 30 - (time.time() - start_time)
    if remaining_time > 5:
        best_result = _local_search(best_result, remaining_time)
    
    return best_result
# EVOLVE-BLOCK-END
