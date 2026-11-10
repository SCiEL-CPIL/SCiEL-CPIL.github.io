"""Rewritten program: exploit affine invariance, fix 3 points, local search on 13."""
import time
import random
import math
from itertools import combinations

def _line_key(p1, p2):
    """Return canonical line representation through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx, dy = -dx, -dy
    c = dx * y1 - dy * x1
    g2 = math.gcd(abs(dx), abs(dy), abs(c))
    if g2 > 1:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _build_line_map(points):
    """Build dict: line_key -> frozenset of point indices on that line."""
    n = len(points)
    lm = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lm:
                lm[key] = set()
            lm[key].add(i)
            lm[key].add(j)
    return lm

def _score_from_line_map(lm):
    """Count lines with exactly 4 points."""
    return sum(1 for s in lm.values() if len(s) == 4)

def _max_collinear(lm):
    if not lm:
        return 0
    return max(len(s) for s in lm.values())

def _is_valid(points):
    """Check distinct, in bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    lm = _build_line_map(points)
    return _max_collinear(lm) <= 4

def _score(points):
    """Return number of exact-4 lines, or -1 if invalid."""
    if len(points) != 16:
        return -1
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return -1
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
    lm = _build_line_map(points)
    if _max_collinear(lm) >= 5:
        return -1
    return _score_from_line_map(lm)

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _delta_score_move(points, idx, new_point, lm):
    """
    Compute the change in score (and max collinear) when moving point idx to new_point.
    Returns (delta_score, new_max_collinear) or None if invalid (duplicate or out of bounds).
    """
    x, y = new_point
    if not isinstance(x, int) or not isinstance(y, int):
        return None
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinctness
    for j, p in enumerate(points):
        if j != idx and (p[0], p[1]) == (x, y):
            return None
    
    old_point = points[idx]
    points[idx] = new_point
    new_lm = _build_line_map(points)
    new_max = _max_collinear(new_lm)
    if new_max >= 5:
        points[idx] = old_point
        return None
    new_score = _score_from_line_map(new_lm)
    old_score = _score_from_line_map(lm)
    points[idx] = old_point
    return (new_score - old_score, new_lm, new_score)

def _local_search(points, max_iters=10000, deadline=None, temp_start=50.0, temp_end=0.5):
    """
    Local search with simulated annealing.
    """
    if deadline is None:
        deadline = time.time() + 60
    
    current = [list(p) for p in points]
    current_lm = _build_line_map(current)
    current_score = _score_from_line_map(current_lm)
    if _max_collinear(current_lm) >= 5:
        return current, -1
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        progress = it / max_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        
        # Generate candidates: random + structured
        candidates = []
        
        # Random candidates
        for _ in range(8):
            p = _random_point()
            if p != old_point:
                candidates.append(p)
        
        # Try to complete 3-point lines: for each line with 3 points not including idx,
        # find a 4th point that would create a 4-line
        for line, idx_set in current_lm.items():
            if len(idx_set) == 3 and idx not in idx_set:
                a, b, c = line
                # Find integer points on this line within bounds
                # a*x + b*y = c
                found = 0
                if a != 0:
                    # Sample y values
                    for y in range(-50, 51):
                        num = c - b * y
                        if num % a == 0:
                            x = num // a
                            if -1000 <= x <= 1000:
                                p = [x, y]
                                if p != old_point:
                                    candidates.append(p)
                                found += 1
                                if found >= 5:
                                    break
                elif b != 0:
                    for x in range(-50, 51):
                        num = c - a * x
                        if num % b == 0:
                            y = num // b
                            if -1000 <= y <= 1000:
                                p = [x, y]
                                if p != old_point:
                                    candidates.append(p)
                                found += 1
                                if found >= 5:
                                    break
        
        if not candidates:
            continue
        
        random.shuffle(candidates)
        
        best_cand = None
        best_cand_score = current_score
        best_cand_lm = None
        
        for cand in candidates[:12]:
            result = _delta_score_move(current, idx, cand, current_lm)
            if result is None:
                continue
            delta, new_lm, new_score = result
            
            if new_score > best_cand_score:
                best_cand_score = new_score
                best_cand = cand
                best_cand_lm = new_lm
            elif new_score < best_cand_score:
                delta_val = new_score - best_cand_score
                if delta_val < 0:
                    prob = math.exp(delta_val / max(temp, 0.01))
                    if random.random() < prob:
                        best_cand_score = new_score
                        best_cand = cand
                        best_cand_lm = new_lm
        
        if best_cand is not None:
            current[idx] = best_cand
            current_score = best_cand_score
            current_lm = best_cand_lm
            if current_score > best_score:
                best_score = current_score
                best = [list(p) for p in current]
        else:
            current[idx] = old_point
    
    return best, best_score

def _generate_grid_4x4():
    """4x4 grid from (0,0) to (3,3)."""
    return [[i, j] for i in range(4) for j in range(4)]

def _generate_perturbed_grid():
    """Perturbed 4x4 grid."""
    pts = _generate_grid_4x4()
    for _ in range(4):
        idx = random.randint(0, 15)
        pts[idx] = [pts[idx][0] + random.randint(-2, 2), pts[idx][1] + random.randint(-2, 2)]
    # Ensure distinct
    seen = set()
    unique = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    while len(unique) < 16:
        p = _random_point()
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    return unique[:16]

def _generate_lattice_structure():
    """Lattice-based structure: mix of grid and diagonal points."""
    pts = []
    # Base 4x4 grid
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    # Add some diagonal perturbations
    for _ in range(3):
        idx = random.randint(0, 15)
        pts[idx] = [pts[idx][0] + random.randint(-3, 3), pts[idx][1] + random.randint(-3, 3)]
    seen = set()
    unique = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    while len(unique) < 16:
        p = _random_point()
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    return unique[:16]

def _generate_random_config():
    pts = []
    seen = set()
    while len(pts) < 16:
        p = _random_point()
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            pts.append(p)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_points = None
    best_score = -1
    
    # Phase 1: Start with 4x4 grid (known good: 10 four-point lines)
    grid_pts = _generate_grid_4x4()
    s = _score(grid_pts)
    if s > best_score:
        best_score = s
        best_points = [list(p) for p in grid_pts]
    
    # Phase 2: Local search from grid
    if time.time() < deadline:
        improved, imp_score = _local_search(best_points, max_iters=15000, deadline=deadline, temp_start=50.0, temp_end=0.5)
        if imp_score > best_score:
            best_score = imp_score
            best_points = [list(p) for p in improved]
    
    # Phase 3: Multiple restarts with diverse starts
    restart_configs = []
    
    # Perturbed grids
    for _ in range(5):
        restart_configs.append(_generate_perturbed_grid())
    
    # Lattice structures
    for _ in range(5):
        restart_configs.append(_generate_lattice_structure())
    
    # Random configs
    for _ in range(5):
        restart_configs.append(_generate_random_config())
    
    random.shuffle(restart_configs)
    
    for i, start_config in enumerate(restart_configs):
        if time.time() >= deadline:
            break
        
        # Skip if invalid
        if _score(start_config) < 0:
            continue
        
        improved, imp_score = _local_search(start_config, max_iters=8000, deadline=deadline, temp_start=30.0, temp_end=0.3)
        if imp_score > best_score:
            best_score = imp_score
            best_points = [list(p) for p in improved]
        
        # If we're doing well, do another round of refinement
        if best_score >= 12 and time.time() < deadline:
            improved2, imp_score2 = _local_search(best_points, max_iters=5000, deadline=deadline, temp_start=20.0, temp_end=0.2)
            if imp_score2 > best_score:
                best_score = imp_score2
                best_points = [list(p) for p in improved2]
    
    # Final validation
    if best_points is None or not _is_valid(best_points):
        return _generate_grid_4x4()
    
    return best_points
