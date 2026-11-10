"""Minimize ordinary lines for 12 integer points via local search."""
import time
import random
import math
from itertools import combinations

_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    # Normalize: make dx > 0, reduce by gcd
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with k points contributes to C(k,2) pairs.
    # Ordinary lines are those with exactly 2 points (count == 1 pair).
    ordinary = sum(1 for cnt in line_counts.values() if cnt == 1)
    return ordinary

def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Not all collinear: check if all points lie on a single line
    # Pick first two distinct points, check if all others are on that line
    p0 = points[0]
    p1 = points[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    all_collinear = True
    for i in range(2, 12):
        p = points[i]
        # Cross product should be 0 if collinear
        if dx * (p[1] - p0[1]) - dy * (p[0] - p0[0]) != 0:
            all_collinear = False
            break
    if all_collinear:
        return False
    return True

def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def _random_points():
    """Generate 12 random integer points."""
    points = []
    seen = set()
    while len(points) < 12:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    return points

def _perturb(points, rng):
    """Perturb one point by a small random amount."""
    new_points = [list(p) for p in points]
    idx = rng.randint(0, 11)
    step = rng.randint(-5, 5)
    if step == 0:
        step = rng.choice([-1, 1])
    axis = rng.randint(0, 1)
    new_points[idx][axis] = _clamp(new_points[idx][axis] + step, -1000, 1000)
    # Ensure distinctness
    for i in range(12):
        for j in range(i + 1, 12):
            if new_points[i] == new_points[j]:
                # Nudge one point
                new_points[j][0] = _clamp(new_points[j][0] + 1, -1000, 1000)
    return new_points

def construct():
    rng = random.Random(42)
    
    # Start with a grid configuration
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Try some known good configurations first
    # Regular polygon-like configurations
    candidates = [
        [[x, y] for x in range(4) for y in range(3)],  # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],  # 3x4 grid
        [[x, y] for x in range(5) if x != 2 for y in range(3)][:12],  # 5x3 minus middle
    ]
    
    for cand in candidates:
        if len(cand) == 12 and _is_valid(cand):
            s = count_ordinary_lines(cand)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in cand]
    
    # Local search
    current_points = [list(p) for p in best_points]
    current_score = best_score
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < _DEADLINE and no_improve < max_no_improve:
        new_points = _perturb(current_points, rng)
        if not _is_valid(new_points):
            no_improve += 1
            continue
        new_score = count_ordinary_lines(new_points)
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_points = [list(p) for p in current_points]
        elif new_score == current_score and rng.random() < 0.1:
            # Occasional sideways move to escape local minima
            current_points = new_points
            no_improve += 1
        else:
            no_improve += 1
        
        # Every 1000 iterations, try a bigger perturbation
        if no_improve % 1000 == 0 and no_improve > 0:
            new_points = [list(p) for p in current_points]
            for _ in range(3):
                idx = rng.randint(0, 11)
                step = rng.randint(-20, 20)
                axis = rng.randint(0, 1)
                new_points[idx][axis] = _clamp(new_points[idx][axis] + step, -1000, 1000)
            if _is_valid(new_points):
                s = count_ordinary_lines(new_points)
                if s <= current_score:
                    current_points = new_points
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_points = [list(p) for p in current_points]
    
    # Return best found
    return best_points
# EVOLVE-BLOCK-END
