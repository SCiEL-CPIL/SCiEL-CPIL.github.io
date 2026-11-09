"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a canonical key for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Canonical direction: make first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal form: line is a*x + b*y = c where (a,b) is normal to (dx,dy)
    # Normal vector: (-dy, dx) or (dy, -dx). Use (dy, -dx) normalized.
    a = dy
    b = -dx
    # Ensure canonical sign for (a,b)
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    return (a, b, c)


def _count_exact4(points):
    """Count lines containing exactly 4 of the given points, and check no 5+ collinear.
    Returns (exact4_count, max_collinear)."""
    n = len(points)
    line_map = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].append((i, j))
    
    # For each line key, we need the set of points on it
    # Rebuild: for each line, collect unique point indices
    point_to_lines = defaultdict(set)
    for key, pairs in line_map.items():
        for (i, j) in pairs:
            point_to_lines[i].add(key)
            point_to_lines[j].add(key)
    
    # Now for each line, count points
    exact4 = 0
    max_collinear = 0
    # Rebuild line -> set of points
    line_points = defaultdict(set)
    for i in range(n):
        for key in point_to_lines[i]:
            line_points[key].add(i)
    
    for key, pts in line_points.items():
        cnt = len(pts)
        if cnt > max_collinear:
            max_collinear = cnt
        if cnt == 4:
            exact4 += 1
    
    return exact4, max_collinear


def _fast_score(points):
    """Return (exact4_count, max_collinear) efficiently."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_points[key].add(i)
            line_points[key].add(j)
    exact4 = 0
    max_collinear = 0
    for key, pts in line_points.items():
        cnt = len(pts)
        if cnt > max_collinear:
            max_collinear = cnt
        if cnt == 4:
            exact4 += 1
    return exact4, max_collinear


def construct():
    random.seed(42)
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start from 4x4 grid (known: 10 exact-4 lines, max_collinear=4)
    base = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in base]
    best_score, best_max = _fast_score(best_points)
    
    if best_max > 4:
        # Shouldn't happen for grid, but guard
        pass
    
    # Hill climb with simulated annealing
    current = [p[:] for p in base]
    cur_score, cur_max = _fast_score(current)
    
    # Precompute all pairs' line keys for fast incremental update
    # Actually, let's just do full re-evaluation; 16 points = 120 pairs, fast enough
    
    def _evaluate(points):
        """Evaluate a configuration. Returns (score, is_valid)."""
        exact4, mx = _fast_score(points)
        return exact4, mx
    
    # Try improving moves
    attempts = 0
    max_attempts = 200000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        # Try several candidate new positions
        found_improvement = False
        for _ in range(20):
            # Random new position within bounds
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            if (new_x, new_y) == (old_x, old_y):
                continue
            # Check distinctness
            if any(current[k][0] == new_x and current[k][1] == new_y for k in range(16) if k != idx):
                continue
            
            current[idx] = [new_x, new_y]
            new_score, new_max = _fast_score(current)
            
            if new_max <= 4 and new_score > cur_score:
                cur_score = new_score
                found_improvement = True
                break
            elif new_max <= 4 and new_score == cur_score:
                # Accept with some probability (annealing)
                if random.random() < 0.1:
                    cur_score = new_score
                    found_improvement = True
                    break
            else:
                current[idx] = [old_x, old_y]
        
        if not found_improvement:
            current[idx] = [old_x, old_y]
        
        if cur_score > best_score and cur_max <= 4:
            best_score = cur_score
            best_max = cur_max
            best_points = [p[:] for p in current]
        
        # Occasionally restart from a perturbed version of best
        if attempts % 5000 == 0:
            current = [p[:] for p in best_points]
            # Perturb a few points
            for _ in range(random.randint(1, 3)):
                k = random.randint(0, 15)
                current[k] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            cur_score, cur_max = _fast_score(current)
            if cur_max > 4:
                current = [p[:] for p in best_points]
                cur_score, cur_max = _fast_score(current)
    
    # Final validation
    final_score, final_max = _fast_score(best_points)
    if final_max > 4:
        # Fallback to grid
        return base
    
    # Ensure all points are within bounds and distinct
    seen = set()
    for p in best_points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return base
        if (x, y) in seen:
            return base
        seen.add((x, y))
    
    if len(best_points) != 16:
        return base
    
    return best_points

# EVOLVE-BLOCK-END
