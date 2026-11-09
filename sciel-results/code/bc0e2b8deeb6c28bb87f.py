import time
import sys
import random
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _hexagon_points():
    """Generate 12 points from a hexagonal lattice pattern."""
    pts = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2], [1, -2]
    ]
    return pts[:12]

def _snap_to_line(points, i, j, k):
    """Try to snap point i onto the line through points j and k.
    Returns the new position if it's a valid integer point on the line, else None."""
    x1, y1 = points[j]
    x2, y2 = points[k]
    dx = x2 - x1
    dy = y2 - y1
    
    # Check if point i can be placed on this line at an integer coordinate
    # The line parametric form: (x1 + t*dx, y1 + t*dy) for integer t
    # We need to find t such that the new point is distinct from all other points
    
    # Try several values of t
    existing = set(tuple(p) for p in points)
    existing.discard(tuple(points[i]))
    
    for t in range(-5, 6):
        if t == 0:
            continue  # This would be point j's position
        nx = x1 + t * dx
        ny = y1 + t * dy
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        if (nx, ny) not in existing:
            # Check it's not the same as current point i
            if (nx, ny) != tuple(points[i]):
                return [nx, ny]
    return None

def _find_line_through(points, i, j):
    """Find the line through points i and j, return (a,b,c) normalized."""
    x1, y1 = points[i]
    x2, y2 = points[j]
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _simulated_annealing(points, max_time=100):
    """Simulated annealing with line-snapping moves to minimize ordinary lines."""
    start = time.time()
    best = [list(p) for p in points]
    best_count = _count_ordinary_lines(best)
    current = [list(p) for p in points]
    current_count = best_count
    
    # Temperature schedule
    T0 = 5.0
    T_final = 0.01
    cooling = 0.995
    
    rng = random.Random(42)
    
    while (time.time() - start) < max_time:
        T = max(T_final, T0 * (cooling ** int((time.time() - start) * 10)))
        
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Snap a point onto a line through two other points
            i = rng.randint(0, 11)
            j, k = rng.sample([x for x in range(12) if x != i], 2)
            new_pos = _snap_to_line(current, i, j, k)
            if new_pos is None:
                # Try a small random displacement
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_pos = [current[i][0] + dx, current[i][1] + dy]
                if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
                    continue
                if (new_pos[0], new_pos[1]) in set(tuple(p) for p in current):
                    continue
            new_points = [list(p) for p in current]
            new_points[i] = new_pos
            new_count = _count_ordinary_lines(new_points)
            
        elif move_type < 0.7:
            # Move one point randomly
            i = rng.randint(0, 11)
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_pos = [current[i][0] + dx, current[i][1] + dy]
            if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
                continue
            if (new_pos[0], new_pos[1]) in set(tuple(p) for p in current):
                continue
            new_points = [list(p) for p in current]
            new_points[i] = new_pos
            new_count = _count_ordinary_lines(new_points)
            
        else:
            # Swap two points' positions (perturbation)
            i, j = rng.sample(range(12), 2)
            new_points = [list(p) for p in current]
            new_points[i], new_points[j] = new_points[j], new_points[i]
            if len(set(tuple(p) for p in new_points)) != 12:
                continue
            new_count = _count_ordinary_lines(new_points)
        
        delta = new_count - current_count
        
        if delta <= 0 or rng.random() < (10 ** (-delta / T)):
            current = new_points
            current_count = new_count
            if current_count < best_count:
                best = [list(p) for p in current]
                best_count = current_count
        
        T *= cooling
    
    return best, best_count

def _greedy_local_search(points, max_time=60):
    """Greedy local search: try moving each point to a nearby integer position
    that reduces the number of ordinary lines."""
    start = time.time()
    best = [list(p) for p in points]
    best_count = _count_ordinary_lines(best)
    
    moves = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx != 0 or dy != 0:
                moves.append((dx, dy))
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for i in range(12):
            if (time.time() - start) > max_time:
                break
            for dx, dy in moves:
                if (time.time() - start) > max_time:
                    break
                new_pts = [list(p) for p in best]
                new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
                
                if len(set(tuple(p) for p in new_pts)) != 12:
                    continue
                if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in new_pts):
                    continue
                
                count = _count_ordinary_lines(new_pts)
                if count < best_count:
                    best = new_pts
                    best_count = count
                    improved = True
                    break
    
    return best, best_count

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses simulated annealing with line-snapping heuristic.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with hexagonal lattice configuration
    init_points = _hexagon_points()
    init_count = _try_config(init_points)
    if init_count is None:
        init_points = [[x, y] for x in range(4) for y in range(3)]
        init_count = _try_config(init_points)
    
    best_points = [list(p) for p in init_points]
    best_count = init_count
    
    # Run simulated annealing with line snapping
    if time.time() < deadline - 10:
        sa_time = min(80, deadline - time.time() - 10)
        sa_points, sa_count = _simulated_annealing(best_points, max_time=sa_time)
        if sa_count < best_count:
            best_points = sa_points
            best_count = sa_count
    
    # Run greedy local search to polish
    if time.time() < deadline - 10:
        ls_time = min(30, deadline - time.time() - 5)
        ls_points, ls_count = _greedy_local_search(best_points, max_time=ls_time)
        if ls_count < best_count:
            best_points = ls_points
            best_count = ls_count
    
    # Try additional starting configurations
    additional_starts = [
        [[x, y] for x in range(3) for y in range(4)],
        [[x, y] for x in range(4) for y in range(3)],
    ]
    
    for pts in additional_starts:
        if time.time() > deadline - 15:
            break
        cnt = _try_config(pts)
        if cnt is not None and cnt < best_count:
            best_points = [list(p) for p in pts]
            best_count = cnt
        if time.time() < deadline - 15:
            sp, sc = _simulated_annealing(pts, max_time=min(20, deadline - time.time() - 5))
            if sc < best_count:
                best_points = sp
                best_count = sc
    
    # Final validation
    final_count = _try_config(best_points)
    if final_count is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
