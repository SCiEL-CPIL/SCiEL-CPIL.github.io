import time
import random
from itertools import combinations
from collections import Counter

def _score(points):
    """Return the maximum multiplicity of any single positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    random.seed(42)
    
    # Initial: 5x6 grid
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = _score(best_points)
    
    # Try different grid arrangements and scales
    for scale in range(1, 11):
        for cols in range(3, 11):
            for rows in range(3, 11):
                if cols * rows == 30:
                    if (cols-1)*scale > 30 or (rows-1)*scale > 30:
                        continue
                    pts = [(x*scale, y*scale) for x in range(cols) for y in range(rows)]
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts[:]
    
    # Local search
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try moving a single point
        for i in range(30):
            if time.time() > deadline:
                break
            for _ in range(30):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) in best_points:
                    continue
                new_points = best_points[:]
                new_points[i] = (nx, ny)
                s = _score(new_points)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    improved = True
                    break
        
        # Try swapping two points
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, 29)
            j = random.randint(0, 29)
            if i == j:
                continue
            new_points = best_points[:]
            new_points[i], new_points[j] = new_points[j], new_points[i]
            s = _score(new_points)
            if s > best_score:
                best_score = s
                best_points = new_points
                improved = True
        
        if not improved and iteration > 100:
            # Random restart
            for _ in range(20):
                if time.time() > deadline:
                    break
                seen = set()
                unique_pts = []
                while len(unique_pts) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        unique_pts.append(p)
                s = _score(unique_pts)
                if s > best_score:
                    best_score = s
                    best_points = unique_pts[:]
    
    return best_points
# EVOLVE-BLOCK-END
