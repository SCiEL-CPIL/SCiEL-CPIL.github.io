"""
Rewritten program using a structural change:
- Use known optimal configurations (e.g., 2x6 grid, 3x4 grid) as starting points
- Implement a more targeted local search that focuses on creating 3+ point lines
- Use a combinatorial approach to identify and exploit collinearity patterns
"""
import time
from math import gcd
from itertools import combinations
import random

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _try_move(pts, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y), return new pts if valid."""
    if new_x == pts[idx][0] and new_y == pts[idx][1]:
        return None
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    for i, p in enumerate(pts):
        if i != idx and p[0] == new_x and p[1] == new_y:
            return None
    new_pts = [list(p) for p in pts]
    new_pts[idx] = [new_x, new_y]
    return new_pts

def _generate_grid(w, h):
    """Generate a w x h grid of points."""
    return [[x, y] for x in range(w) for y in range(h)]

def _generate_2x6():
    """Generate 2x6 grid - known to have good collinearity properties."""
    return [[x, y] for x in range(6) for y in range(2)]

def _generate_3x4():
    """Generate 3x4 grid."""
    return [[x, y] for x in range(3) for y in range(4)]

def _generate_4x3():
    """Generate 4x3 grid."""
    return [[x, y] for x in range(4) for y in range(3)]

def _generate_diagonal_config():
    """Generate a configuration with diagonal lines for better collinearity."""
    # Points arranged to create multiple 3+ point lines
    return [
        [0, 0], [1, 1], [2, 2], [3, 3],  # diagonal
        [0, 1], [1, 0], [2, 0], [3, 0],  # mixed
        [0, 2], [1, 2], [2, 1], [3, 2]   # mixed
    ]

def _generate_hexagonal():
    """Generate hexagonal-like configuration."""
    return [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3]
    ]

def _local_search_optimized(pts, deadline):
    """Optimized local search focusing on creating 3+ point lines."""
    best_pts = [list(p) for p in pts]
    best_score = _count_ordinary(best_pts)
    n = len(best_pts)
    
    # Precompute line structure
    def get_line_structure(points):
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map
    
    # Targeted moves: try to create 3+ point lines
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        line_map = get_line_structure(best_pts)
        ordinary_lines = [key for key, v in line_map.items() if len(v) == 2]
        
        if not ordinary_lines:
            break
        
        # For each ordinary line, try to move one of its points to create a 3+ line
        for line_key in ordinary_lines:
            if time.time() > deadline:
                break
                
            a, b, c = line_key
            line_pts = line_map[line_key]
            
            for idx in line_pts:
                orig_x, orig_y = best_pts[idx]
                
                # Try to place point on a line with 2+ other points
                for other_key, other_pts in line_map.items():
                    if other_key == line_key:
                        continue
                    if len(other_pts) < 2:
                        continue
                    
                    oa, ob, oc = other_key
                    # Check if moving idx to a point on this line helps
                    # Try a few candidate positions on the line
                    if ob != 0:
                        for _ in range(3):
                            x_new = random.randint(-50, 50)
                            num = -oa * x_new - oc
                            if num % ob == 0:
                                y_new = num // ob
                                if abs(x_new) <= 10**9 and abs(y_new) <= 10**9:
                                    new_pts = _try_move(best_pts, idx, x_new, y_new)
                                    if new_pts:
                                        score = _count_ordinary(new_pts)
                                        if score < best_score:
                                            best_pts = new_pts
                                            best_score = score
                                            break
                    
                    # Also try small perturbations
                    for _ in range(5):
                        dx = random.randint(-3, 3)
                        dy = random.randint(-3, 3)
                        new_pts = _try_move(best_pts, idx, orig_x + dx, orig_y + dy)
                        if new_pts:
                            score = _count_ordinary(new_pts)
                            if score < best_score:
                                best_pts = new_pts
                                best_score = score
                                break
    
    return best_pts, best_score

def construct():
    deadline = time.time() + 140
    
    # Start with multiple known good configurations
    initial_configs = [
        _generate_2x6(),
        _generate_3x4(),
        _generate_4x3(),
        _generate_diagonal_config(),
        _generate_hexagonal(),
    ]
    
    best_pts = None
    best_score = float('inf')
    
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        best_pts = _generate_2x6()
        best_score = _count_ordinary(best_pts)
    
    random.seed(42)
    
    # Run local search from best initial config
    best_pts, best_score = _local_search_optimized(best_pts, deadline)
    
    # Try additional random restarts if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Random restart
            random_pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(12)]
            if not _is_valid(random_pts):
                continue
            score = _count_ordinary(random_pts)
            if score < best_score:
                best_pts, best_score = _local_search_optimized(random_pts, deadline)
    
    return [list(p) for p in best_pts]
