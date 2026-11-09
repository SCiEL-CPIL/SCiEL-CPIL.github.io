import time
import sys
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Return canonical (a,b,c) for line through two points."""
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

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            line_map[key].add(i)
            line_map[key].add(j)
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _build_line_structure(points):
    """Build line_map and point_lines for incremental updates."""
    n = len(points)
    line_map = defaultdict(set)
    point_lines = defaultdict(set)  # point index -> set of line keys
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            line_map[key].add(i)
            line_map[key].add(j)
            point_lines[i].add(key)
            point_lines[j].add(key)
    return line_map, point_lines

def _count_ordinary_from_structure(line_map):
    return sum(1 for pts in line_map.values() if len(pts) == 2)

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

def _simulated_annealing(points, max_time=60, seed=None):
    """Simulated annealing with incremental line-count updates."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    best = [list(p) for p in points]
    best_count = _count_ordinary_lines(best)
    
    # Build initial structure
    line_map, point_lines = _build_line_structure(best)
    current_count = best_count
    
    # Temperature schedule
    T0 = 10.0
    T_end = 0.1
    n_points = len(best)
    
    improved = True
    no_improve_steps = 0
    max_no_improve = 5000
    
    while (time.time() - start) < max_time and no_improve_steps < max_no_improve:
        # Choose a random point to move
        i = random.randint(0, n_points - 1)
        
        # Choose a random displacement (small moves for fine-tuning)
        if random.random() < 0.7:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        else:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
        
        if dx == 0 and dy == 0:
            continue
        
        new_x = best[i][0] + dx
        new_y = best[i][1] + dy
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        
        # Check collision
        new_pt = (new_x, new_y)
        collides = False
        for j in range(n_points):
            if j != i and (best[j][0], best[j][1]) == new_pt:
                collides = True
                break
        if collides:
            continue
        
        # Compute delta: remove old lines involving point i, add new lines
        # Old lines involving i
        old_lines = point_lines[i].copy()
        
        # New lines involving i
        new_lines = set()
        for j in range(n_points):
            if j != i:
                key = _line_key(new_x, new_y, best[j][0], best[j][1])
                new_lines.add(key)
        
        # Compute change in ordinary line count
        # For each line that changes membership, track if it becomes/ceases to be ordinary
        
        # Lines that lose point i (old lines not in new_lines)
        # Lines that gain point i (new lines not in old_lines)
        # Lines that keep point i (intersection) - their size doesn't change
        
        lost_lines = old_lines - new_lines
        gained_lines = new_lines - old_lines
        kept_lines = old_lines & new_lines
        
        delta = 0
        
        # For lost lines: point i leaves. If line had exactly 2 points, it becomes 1 (not a line anymore, but we track by point sets)
        # Actually, a line with 1 point is not counted. A line with 2 points is ordinary.
        # When point i leaves a line:
        #   - If line had 2 points (i and one other), it now has 1 point -> was ordinary, no longer exists as a 2+ line. Delta: -1
        #   - If line had 3+ points, it now has 2+ points. If it had exactly 3, it now has 2 -> becomes ordinary. Delta: +1
        for key in lost_lines:
            if key in line_map:
                sz = len(line_map[key])
                if sz == 2:
                    # Was ordinary, now has 1 point (remove from map)
                    delta -= 1
                elif sz == 3:
                    # Was not ordinary (3 pts), now has 2 -> ordinary
                    delta += 1
                # sz > 3: still not ordinary, no change
            # If key not in line_map, it shouldn't happen since we're removing from it
        
        # For gained lines: point i joins.
        #   - If line didn't exist (0 points), it now has 1 point -> not ordinary. Delta: 0
        #   - If line had 1 point, it now has 2 -> ordinary. Delta: +1
        #   - If line had 2 points, it now has 3 -> was ordinary, no longer. Delta: -1
        #   - If line had 3+ points, no change in ordinary status.
        for key in gained_lines:
            if key in line_map:
                sz = len(line_map[key])
                if sz == 1:
                    delta += 1
                elif sz == 2:
                    delta -= 1
                # sz >= 3: no change
            else:
                # Line didn't exist, now has 1 point: not ordinary
                delta += 0
        
        # Apply the move to the data structures
        old_x, old_y = best[i][0], best[i][1]
        
        # Remove point i from old lines
        for key in lost_lines:
            if key in line_map:
                line_map[key].discard(i)
                if len(line_map[key]) == 0:
                    del line_map[key]
        # Update point_lines for other points on lost lines
        for key in lost_lines:
            if key in point_lines:
                # Remove key from other points' line sets
                pass  # We'll rebuild point_lines at the end or track carefully
        
        # Actually, let's just update point_lines for all affected points
        # For simplicity, let's do a full rebuild of point_lines after the move
        # But that's O(n^2). Let's be smarter.
        
        # Let me redo this more carefully.
        # I'll update line_map and then rebuild point_lines from scratch (O(n^2) but n=12 so it's fine)
        
        # Remove point i from all its old lines in line_map
        for key in old_lines:
            if key in line_map:
                line_map[key].discard(i)
                if len(line_map[key]) == 0:
                    del line_map[key]
        
        # Add point i to all its new lines in line_map
        for key in new_lines:
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
        
        # Update the actual point position
        best[i][0] = new_x
        best[i][1] = new_y
        
        # Rebuild point_lines (n=12, so O(n^2) is cheap)
        point_lines = defaultdict(set)
        for i2 in range(n_points):
            for j2 in range(i2 + 1, n_points):
                key = _line_key(best[i2][0], best[i2][1], best[j2][0], best[j2][1])
                point_lines[i2].add(key)
                point_lines[j2].add(key)
        
        current_count += delta
        
        # Accept or reject
        if current_count < best_count:
            best_count = current_count
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            # Metropolis acceptance
            if random.random() < random.expovariate(1.0 / max(T0 * (T_end / T0) ** (no_improve_steps / 5000), 0.01)):
                pass  # Accept
            else:
                # Reject: undo the move
                best[i][0] = old_x
                best[i][1] = old_y
                current_count -= delta
                # Rebuild structures
                line_map, point_lines = _build_line_structure(best)
        
        # Update temperature based on steps
        # Simple cooling: T decreases over time
        elapsed = time.time() - start
        if elapsed > 0:
            T = T0 * (T_end / T0) ** (elapsed / max_time)
    
    # Final validation
    final_count = _count_ordinary_lines(best)
    if final_count < best_count:
        best_count = final_count
    
    return best, best_count

def _hexagon_points():
    """Generate 12 points from a hexagonal lattice pattern."""
    pts = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2]
    ]
    return pts

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses simulated annealing with incremental line-count updates.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Multiple diverse starting configurations
    starts = [
        _hexagon_points(),
        [[x, y] for x in range(4) for y in range(3)],  # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],  # 3x4 grid
        # Another hexagon variant
        [
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 2], [-1, 2], [-2, 0], [1, -2]
        ],
        # Random-ish structured start
        [[i, (i * 7) % 5 - 2] for i in range(12)],
    ]
    
    best_points = None
    best_count = float('inf')
    
    # Try each start with SA
    n_starts = len(starts)
    time_per_start = (deadline - start_time - 5) / n_starts
    
    for idx, init_pts in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        
        # Validate
        cnt = _try_config(init_pts)
        if cnt is None:
            continue
        
        if cnt < best_count:
            best_points = [list(p) for p in init_pts]
            best_count = cnt
        
        # Run SA with time budget
        sa_time = min(time_per_start, remaining - 2)
        if sa_time < 3:
            sa_time = 3
        
        sa_pts, sa_cnt = _simulated_annealing(init_pts, max_time=sa_time, seed=42 + idx)
        
        if sa_cnt < best_count:
            best_points = sa_pts
            best_count = sa_cnt
    
    # If we still have time, do another round of SA from the best found
    if time.time() < deadline - 15 and best_points is not None:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            sa_pts, sa_cnt = _simulated_annealing(best_points, max_time=remaining, seed=999)
            if sa_cnt < best_count:
                best_points = sa_pts
                best_count = sa_cnt
    
    # Final validation
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    final_count = _try_config(best_points)
    if final_count is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        final_count = _try_config(best_points)
    
    return best_points
