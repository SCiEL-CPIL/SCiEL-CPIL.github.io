"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _doubled_areas(points):
    """Compute all positive doubled areas for triples of points."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    # pts shape: (n, 2)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # doubled area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
                det = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + 
                         pts[j,0]*(pts[k,1]-pts[i,1]) + 
                         pts[k,0]*(pts[i,1]-pts[j,1]))
                if det > 0:
                    areas.append(det)
    return areas

def _best_multiplicity(points):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _evaluate(points):
    """Return (multiplicity, area) of the best doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return (0, 0)
    c = Counter(areas)
    best_area, best_mult = max(c.items(), key=lambda x: x[1])
    return (best_mult, best_area)

def _hill_climb(points, max_iters=50000):
    """Hill-climb by moving individual points to nearby cells."""
    best = list(points)
    best_score = _best_multiplicity(best)
    
    # Define neighborhood moves
    def neighbors(p):
        x, y = p
        candidates = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    candidates.append((nx, ny))
        return candidates
    
    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break
        improved = False
        # Try moving each point
        for idx in range(len(best)):
            if _remaining_time() < 5.0:
                break
            orig = best[idx]
            for new_pt in neighbors(orig):
                # Skip if already used
                if new_pt in best:
                    continue
                trial = list(best)
                trial[idx] = new_pt
                score = _best_multiplicity(trial)
                if score > best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def _random_points():
    """Generate 30 random distinct points in [-30,30]^2."""
    import random
    pts = set()
    while len(pts) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        pts.add((x, y))
    return list(pts)

def _grid_points():
    """Generate a 6x5 grid of points."""
    return [(x, y) for x in range(6) for y in range(5)]

def _structured_points():
    """Generate points on a structured pattern that may yield equal areas."""
    # Try a few structured configurations
    patterns = []
    
    # 1. Regular hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 4 + (row % 2) * 2
            y = row * 3
            if -30 <= x <= 30 and -30 <= y <= 30:
                hex_pts.append((x, y))
    if len(hex_pts) == 30:
        patterns.append(hex_pts)
    
    # 2. Square lattice with spacing
    sq_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 5
            y = row * 5
            sq_pts.append((x, y))
    if len(sq_pts) == 30:
        patterns.append(sq_pts)
    
    # 3. Points on a circle-like arrangement
    circle_pts = []
    import math
    angles = np.linspace(0, 2*math.pi, 30, endpoint=False)
    radii = [5, 10, 15, 20, 25]
    for i, angle in enumerate(angles):
        r = radii[i % len(radii)]
        x = int(round(r * math.cos(angle)))
        y = int(round(r * math.sin(angle)))
        if -30 <= x <= 30 and -30 <= y <= 30:
            circle_pts.append((x, y))
    if len(set(circle_pts)) == 30:
        patterns.append(list(set(circle_pts)))
    
    return patterns

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Start with multiple initial configurations
    initial_configs = []
    
    # Grid baseline
    initial_configs.append(_grid_points())
    
    # Structured patterns
    for pat in _structured_points():
        initial_configs.append(pat)
    
    # Random starts
    import random
    random.seed(42)
    for _ in range(5):
        initial_configs.append(_random_points())
    
    best_points = None
    best_score = 0
    
    for config in initial_configs:
        if _remaining_time() < 10.0:
            break
        
        # Validate
        if len(set(config)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in config):
            continue
        
        # Hill-climb from this config
        result, score = _hill_climb(config)
        if score > best_score:
            best_score = score
            best_points = result
    
    # If we still have time, do more random restarts
    if best_points is None:
        best_points = _grid_points()
    
    while _remaining_time() > 15.0:
        random_cfg = _random_points()
        result, score = _hill_climb(random_cfg, max_iters=20000)
        if score > best_score:
            best_score = score
            best_points = result
    
    return best_points

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
