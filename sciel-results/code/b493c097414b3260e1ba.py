"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # For each pair, determine the line, then count how many points are on it
    # Use a dict: line_key -> set of point indices
    # Line through (x1,y1) and (x2,y2): normalize direction
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            # Canonical form: make dx > 0, or dx==0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line key: (dx, dy, x1*dx - y1*dy) ... actually use the line equation
            # ax + by = c where a=dy, b=-dx, c = dy*x1 - dx*y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize the line key
            if a == 0 and b == 0:
                continue
            # Make the key canonical
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
    return ordinary

def _random_point_on_lattice(lo, hi):
    """Generate a random integer point in [lo, hi]^2."""
    return [random.randint(lo, hi), random.randint(lo, hi)]

def _generate_initial_config(n=12, lo=-50, hi=50):
    """Generate a random initial configuration of n distinct points."""
    points = []
    used = set()
    while len(points) < n:
        p = _random_point_on_lattice(lo, hi)
        key = (p[0], p[1])
        if key not in used:
            used.add(key)
            points.append(p)
    return points

def _try_local_search(points, max_iters=50000, lo=-100, hi=100):
    """Local search: try to reduce ordinary lines by moving one point at a time."""
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    n = len(current)
    
    for it in range(max_iters):
        # Pick a random point to move
        idx = random.randint(0, n-1)
        old_point = current[idx][:]
        
        # Try a few random new positions
        improved = False
        for _ in range(5):
            new_point = _random_point_on_lattice(lo, hi)
            # Check distinctness
            if any(new_point[0] == current[i][0] and new_point[1] == current[i][1] 
                   for i in range(n) if i != idx):
                continue
            current[idx] = new_point
            new_score = _count_ordinary_lines(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best = [p[:] for p in current]
                break
            else:
                current[idx] = old_point[:]
        
        if not improved:
            current[idx] = old_point[:]
    
    return best, best_score

def _generate_structured_config():
    """Generate a structured configuration known to have few ordinary lines.
    Based on known constructions: e.g., points on a conic or specific grids."""
    configs = []
    
    # Try a 4x3 grid (same as seed but let's verify)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Try points on a parabola y = x^2 mod something, or specific patterns
    # Known: 12 points can achieve 6 ordinary lines with certain configurations
    
    # Try a configuration based on two lines with 6 points each, plus intersections
    # Or use the Hesse configuration idea
    
    # Let's try: 4 points on x-axis, 4 on y-axis, 4 on diagonal, but avoid all collinear
    # Actually, let's try a few structured patterns
    
    # Pattern 1: 3x4 grid (seed)
    # Pattern 2: 4x3 grid (transpose)
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Pattern 3: Points arranged in a more symmetric way
    # 12 points: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2) is 3x4
    
    # Let's try: remove some points from a larger grid and add others
    # Or try: 6 points on one line, 6 on another, but they intersect
    # If 6 on line L1 and 6 on line L2, and L1 and L2 intersect at 1 point,
    # then we have 11 distinct points. Add one more.
    # Ordinary lines: lines through exactly 2 of these points.
    
    # Let's try a different approach: use points from a small set of lines
    # 4 lines, each with 3-4 points, with controlled intersections
    
    return configs

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_points = None
    best_score = float('inf')
    
    # Try structured configurations first
    for config in _generate_structured_config():
        if len(config) != 12:
            continue
        # Check distinctness
        if len(set((p[0], p[1]) for p in config)) != 12:
            continue
        score = _count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
        if time.time() - start_time > deadline:
            break
    
    # Local search from multiple random starts
    random.seed(42)
    
    num_starts = 20
    for s in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        # Generate initial config
        if s == 0 and best_points is not None:
            points = [p[:] for p in best_points]
        else:
            points = _generate_initial_config(12, lo=-30, hi=30)
        
        # Local search
        found, score = _try_local_search(points, max_iters=20000, lo=-100, hi=100)
        
        if score < best_score:
            best_score = score
            best_points = found
        
        # Check deadline
        if time.time() - start_time > deadline:
            break
    
    # Final check: ensure validity
    if best_points is None:
        # Fallback to seed
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    # Verify: 12 distinct points, not all collinear
    assert len(best_points) == 12
    assert len(set((p[0], p[1]) for p in best_points)) == 12
    # Check not all collinear
    if len(best_points) >= 3:
        (x1, y1), (x2, y2), (x3, y3) = best_points[0], best_points[1], best_points[2]
        if not (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) == 0):
            pass  # Not collinear, good
        else:
            # All three are collinear, check if ALL are
            all_collinear = True
            for i in range(3, 12):
                xi, yi = best_points[i]
                if x1*(y2-yi) + x2*(yi-y1) + xi*(y1-y2) != 0:
                    all_collinear = False
                    break
            if all_collinear:
                # Perturb one point to break collinearity
                best_points[0] = [best_points[0][0] + 1, best_points[0][1]]
    
    return best_points
# EVOLVE-BLOCK-END
