"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - t0
    
    def doubled_areas(points):
        """Return dict: doubled_area -> count (positive only)."""
        cnt = defaultdict(int)
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        cnt[d] += 1
        return cnt
    
    def score_of(points):
        areas = doubled_areas(points)
        if not areas:
            return 0
        return max(areas.values())
    
    # Start with a good initial configuration
    # Try a 6x5 grid
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = score_of(best_points)
    
    # Try some other grid configurations
    configs = [
        [(x, y) for x in range(10) for y in range(3)],
        [(x, y) for x in range(5) for y in range(6)],
        [(x, y) for x in range(15) for y in range(2)],
        [(x, y) for x in range(30) for y in range(1)],
    ]
    
    for config in configs:
        if len(config) != 30:
            continue
        # Check all points are in [-30, 30]
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
            s = score_of(config)
            if s > best_score:
                best_score = s
                best_points = config
    
    # Try some random configurations
    import random
    random.seed(42)
    
    for attempt in range(100):
        if elapsed() > DEADLINE:
            break
        
        points = []
        while len(points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in points:
                points.append((x, y))
        
        s = score_of(points)
        if s > best_score:
            best_score = s
            best_points = points
    
    return best_points
# EVOLVE-BLOCK-END
