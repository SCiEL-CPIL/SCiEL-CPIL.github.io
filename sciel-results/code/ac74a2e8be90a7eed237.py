import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def _count4(pts):
    n = len(pts)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _get_line_key(pts[i], pts[j])
            if k is None:
                continue
            line_map[k] = line_map.get(k, 0) + 1
    c4 = 0
    for v in line_map.values():
        if v >= 5:
            return -1
        if v == 4:
            c4 += 1
    return c4

def _line_intersection(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    x_num = b1 * c2 - b2 * c1
    y_num = a2 * c1 - a1 * c2
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    return (x, y)

def _line_points(l, n_pts, used):
    a, b, c = l
    pts = []
    if a == 0:
        if c % b != 0:
            return pts
        y = -c // b
        if abs(y) > 1000:
            return pts
        for x in range(-1000, 1001):
            if (x, y) not in used:
                pts.append((x, y))
    elif b == 0:
        if c % a != 0:
            return pts
        x = -c // a
        if abs(x) > 1000:
            return pts
        for y in range(-1000, 1001):
            if (x, y) not in used:
                pts.append((x, y))
    else:
        g = gcd(abs(a), abs(b))
        if c % g != 0:
            return pts
        a0 = a // g
        b0 = b // g
        c0 = c // g
        # find one solution
        x0, y0 = _ext_gcd(a0, b0)
        x0 = x0 * c0
        y0 = y0 * c0
        step = b0
        dstep = a0
        # iterate t
        t_min = -10000
        t_max = 10000
        for t in range(t_min, t_max + 1):
            x = x0 + step * t
            y = y0 - dstep * t
            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                if (x, y) not in used:
                    pts.append((x, y))
    return pts

def _ext_gcd(a, b):
    if b == 0:
        return (1, 0)
    x, y = _ext_gcd(b, a % b)
    return (y, x - (a // b) * y)

def _solve_csp(line_groups, time_limit):
    """
    line_groups: list of lists of line keys. Each inner list is a "4-line" that must be realized.
    We try to assign 16 points such that each 4-line has exactly 4 of our points, no 5 collinear.
    Returns points list or None.
    """
    # Flatten: we need to choose 4 points per line, with overlaps at intersections.
    # Strategy: pick the set of all intersection points between lines in the same group as "anchors".
    # Then for each line, we need 4 points: some are anchors, some are free.
    
    # Simpler: use backtracking over points.
    # Order lines by "constraint": lines that share many intersections first.
    
    # Collect all anchor points (intersections of lines within the same group)
    anchors = set()
    line_anchors = defaultdict(set)  # line_key -> set of anchor points on it
    for group in line_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pt = _line_intersection(group[i], group[j])
                if pt is not None:
                    anchors.add(pt)
                    line_anchors[group[i]].add(pt)
                    line_anchors[group[j]].add(pt)
    
    # Also, lines in different groups can intersect; those are potential points too, but we don't force them.
    
    # We'll do a backtracking search:
    # 1. For each line group, we need to pick 4 points on each line.
    # 2. Points can be shared between lines (at intersections).
    # 3. Total distinct points must be 16.
    
    # This is complex. Let's use a simpler heuristic: 
    # Assign points line by line. For each line, pick 4 points. 
    # Prefer anchors first. Then check global constraints.
    
    # Actually, let's just do a randomized local search but guided by the line structure.
    # Initialize: for each line, pick 4 random points on it (preferring intersections).
    # Then run a local search that moves points to satisfy line counts.
    
    points = []
    used = set()
    
    # First, place all anchors
    for pt in anchors:
        if len(points) < 16:
            points.append(pt)
            used.add(pt)
    
    # For each line, ensure it has 4 points
    for group in line_groups:
        for line in group:
            # Count how many points are already on this line
            on_line = [p for p in points if line[0]*p[0] + line[1]*p[1] + line[2] == 0]
            need = 4 - len(on_line)
            if need <= 0:
                continue
            # Find candidate points on this line
            candidates = _line_points(line, 10, used)
            # Pick 'need' candidates, preferring those that are on other lines in the group
            for cand in candidates[:need]:
                if len(points) < 16:
                    points.append(cand)
                    used.add(cand)
    
    # Fill remaining points randomly
    while len(points) < 16:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in used:
            points.append((x, y))
            used.add((x, y))
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = _count4(points)
    
    if best_score < 0:
        return best_points
    
    # Phase 1: Local search (same as before but faster)
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.99998
    current_score = best_score
    
    while time.time() < deadline - 20.0:
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        if random.random() < 0.1:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
        else:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        collision = False
        for k in range(16):
            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                collision = True
                break
        if collision:
            continue
        
        points[idx] = [new_x, new_y]
        new_score = _count4(points)
        
        if new_score >= 0:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                current_score = new_score
            elif new_score == current_score:
                current_score = new_score
            else:
                prob = min(1.0, pow(2.71828, (new_score - current_score) / max(T, T_min)))
                if random.random() < prob:
                    current_score = new_score
                else:
                    points[idx] = [old_x, old_y]
        else:
            points[idx] = [old_x, old_y]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            if random.random() < 0.05:
                T = 1.5
    
    # Phase 2: Directed line-structure search
    # Try to find a configuration with more 4-lines by explicitly constructing line groups.
    # We'll try small line groups (2-4 lines) and see if we can realize them.
    
    # Generate candidate line groups
    candidate_groups = []
    
    # Simple: parallel lines (no intersections)
    for offset in range(-10, 11, 2):
        group = [
            (0, 1, offset),      # y = -offset
            (0, 1, offset + 2),  # y = -offset - 2
            (1, 0, offset),      # x = -offset
            (1, 0, offset + 2)   # x = -offset - 2
        ]
        candidate_groups.append(group)
    
    # Cross: two vertical, two horizontal
    for x1 in range(-10, 11, 3):
        for x2 in range(x1 + 3, 11, 3):
            for y1 in range(-10, 11, 3):
                for y2 in range(y1 + 3, 11, 3):
                    group = [
                        (1, 0, -x1),
                        (1, 0, -x2),
                        (0, 1, -y1),
                        (0, 1, -y2)
                    ]
                    candidate_groups.append(group)
    
    # Diagonal lines
    for slope in [1, -1, 2, -2]:
        for b in range(-10, 11, 2):
            a = -slope
            group = [
                (a, 1, -b),
                (a, 1, -b - 2),
            ]
            if len(group) >= 2:
                candidate_groups.append(group)
    
    # Try each candidate group
    random.shuffle(candidate_groups)
    
    for group in candidate_groups:
        if time.time() > deadline - 5.0:
            break
        
        # Try to realize this group
        pts = _solve_csp([group], deadline - time.time())
        if pts is None:
            continue
        
        # Convert to list of lists
        pts_list = [[p[0], p[1]] for p in pts]
        
        # Validate and score
        score = _count4(pts_list)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts_list]
    
    # Final validation
    final_score = _count4(best_points)
    if final_score < 0:
        grid_pts = [[x, y] for x in range(4) for y in range(4)]
        return grid_pts
    
    return best_points
