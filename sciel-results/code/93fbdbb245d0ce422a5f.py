"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
_DEADLINE = 80.0  # safety margin before 90s kill

def _score(points):
    """Compute max multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all triple determinants vectorized
    # For triples (i, j, k): det = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    best_count = 0
    # Sample-based approach for speed: compute all areas, count frequencies
    # Full C(30,3) = 4060 triples
    areas = []
    idx = np.arange(n)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                d = abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[j,1]-pts[i,1])*(pts[k,0]-pts[i,0]))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _score_fast(points):
    """Faster scoring using numpy vectorization for all triples at once."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Build index arrays for all C(n,3) triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    xi = pts[i_idx, 0]; yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]; yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]; yk = pts[k_idx, 1]
    
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    # Count frequencies
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _gen_random_grid():
    """Generate a random 30-point set in [-30,30]^2."""
    np.random.seed(int(time.time() * 1000) % (2**31))
    while True:
        pts = set()
        while len(pts) < 30:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            pts.add((x, y))
        return list(pts)

def _gen_diagonal():
    """Generate points along diagonal bands."""
    pts = set()
    # Points near y = x
    for x in range(-10, 11):
        for dy in [-2, -1, 0, 1, 2]:
            pts.add((x, x + dy))
            if len(pts) >= 30:
                break
        if len(pts) >= 30:
            break
    # Fill remaining randomly
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _gen_parabola():
    """Generate points along a parabolic curve."""
    pts = set()
    for x in range(-10, 11):
        y = (x * x) // 4 - 25
        for dy in [-1, 0, 1]:
            yy = max(-30, min(30, y + dy))
            pts.add((x, yy))
            if len(pts) >= 30:
                break
        if len(pts) >= 30:
            break
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _gen_lattice():
    """Generate points on a regular grid pattern."""
    pts = set()
    # Try a 6x5 grid centered
    for x in range(-5, 5):
        for y in range(-4, 5):
            pts.add((x, y))
    return list(pts)[:30]

def _gen_spiral():
    """Generate points in a spiral pattern."""
    pts = set()
    # Spiral out from center
    cx, cy = 0, 0
    dx, dy = 1, 0
    steps = 0
    for radius in range(0, 20):
        for _ in range(2):
            for _ in range(radius * 2):
                cx += dx
                cy += dy
                if -30 <= cx <= 30 and -30 <= cy <= 30:
                    pts.add((cx, cy))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
            dx, dy = dy, -dx
        if len(pts) >= 30:
            break
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _local_search(points, time_limit=70.0):
    """Simulated annealing local search."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score_fast(best_points)
    current_points = list(points)
    current_score = best_score
    
    n = len(current_points)
    coord_range = list(range(-30, 31))
    
    temp = 5.0
    cooling = 0.995
    min_temp = 0.01
    
    while time.time() < time_limit and temp > min_temp:
        # Pick a random point to move
        idx = np.random.randint(n)
        old_pt = current_points[idx]
        
        # Random new position
        new_x = np.random.randint(-30, 31)
        new_y = np.random.randint(-30, 31)
        new_pt = (new_x, new_y)
        
        # Check if already taken
        if new_pt in current_points and new_pt != old_pt:
            temp *= cooling
            continue
        
        # Try the move
        current_points[idx] = new_pt
        new_score = _score_fast(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
        else:
            # Accept with probability exp((new_score - current_score) / temp)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.001))
                if np.random.random() > prob:
                    current_points[idx] = old_pt
                    # Revert
                # else: keep the move
            # if delta == 0, keep it (neutral)
        
        temp *= cooling
        
        # Periodic check
        if (time.time() - start_time) > 1.0:
            # Small pause to avoid issues
            pass
    
    return best_points, best_score

def _try_all_starts():
    """Try multiple starting configurations and local search on each."""
    start_time = time.time()
    deadline = time_limit = 75.0  # leave 5s margin
    
    generators = [
        _gen_random_grid,
        _gen_diagonal,
        _gen_parabola,
        _gen_lattice,
        _gen_spiral,
    ]
    
    best_points = None
    best_score = 0
    
    # First, evaluate all initial configurations
    initial_configs = []
    for gen in generators:
        if time.time() > deadline:
            break
        try:
            config = gen()
            if len(set(config)) != 30:
                continue
            s = _score_fast(config)
            initial_configs.append((s, config))
        except Exception:
            pass
    
    # Sort by score descending
    initial_configs.sort(key=lambda x: -x[0])
    
    # Run local search on top configs
    for s, config in initial_configs:
        if time.time() > deadline - 2.0:
            break
        if s > best_score:
            best_score = s
            best_points = list(config)
        
        # Run local search with remaining time
        remaining = deadline - time.time() - 1.0
        if remaining > 2.0:
            try:
                lp, ls = _local_search(config, time_limit=remaining)
                if ls > best_score:
                    best_score = ls
                    best_points = lp
            except Exception:
                pass
    
    # If no good results, try a few more random starts
    if time.time() < deadline - 3.0:
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            try:
                config = _gen_random_grid()
                s = _score_fast(config)
                if s > best_score:
                    best_score = s
                    best_points = list(config)
                remaining = deadline - time.time() - 1.0
                if remaining > 3.0:
                    lp, ls = _local_search(config, time_limit=remaining)
                    if ls > best_score:
                        best_score = ls
                        best_points = lp
            except Exception:
                pass
    
    if best_points is None:
        best_points = _gen_lattice()
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 80.0
    
    try:
        result = _try_all_starts()
    except Exception:
        # Fallback to simple grid
        result = [(x, y) for x in range(6) for y in range(5)]
    
    # Verify validity
    pts = set(result)
    if len(pts) != 30:
        result = [(x, y) for x in range(6) for y in range(5)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(6) for y in range(5)]
            break
    
    return result
# EVOLVE-BLOCK-END
