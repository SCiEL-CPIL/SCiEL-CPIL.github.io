"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def _normalize_line(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
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
    # Ensure consistent sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line: dx*y - dy*x = c
    c = dx * y1 - dy * x1
    g2 = math.gcd(abs(dx), abs(dy))
    if g2 > 1:
        dx //= g2
        dy //= g2
        c //= g2
    return (dx, dy, c)

def _point_on_line(p, line):
    """Check if point p lies on the given line (a, b, c) meaning a*x + b*y = c."""
    a, b, c = line
    x, y = p
    return a * x + b * y == c

def _compute_line_map(points):
    """Compute a map from line -> set of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            line = _normalize_line(points[i], points[j])
            if line not in line_map:
                line_map[line] = set()
            line_map[line].add(i)
            line_map[line].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points from a precomputed line map."""
    count = 0
    for line, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _max_collinear_from_line_map(line_map):
    """Return max number of collinear points from a precomputed line map."""
    if not line_map:
        return 0
    return max(len(s) for s in line_map.values())

def _score(points):
    """Return the number of lines with exactly 4 points, or -1 if invalid."""
    if len(points) != 16:
        return -1
    seen = set()
    for p in points:
        if len(p) != 2:
            return -1
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return -1
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
    
    line_map = _compute_line_map(points)
    if _max_collinear_from_line_map(line_map) >= 5:
        return -1
    return _score_from_line_map(line_map)

def _random_point():
    """Generate a random integer point within bounds."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _generate_grid_config(base=4, offset=0):
    """Generate a grid-like configuration."""
    points = []
    for i in range(base):
        for j in range(base):
            points.append([i + offset, j + offset])
    return points

def _generate_diverse_config():
    """Generate a diverse random configuration."""
    points = []
    attempts = 0
    while len(points) < 16 and attempts < 10000:
        p = _random_point()
        if p not in points:
            points.append(p)
        attempts += 1
    return points

def _fast_local_search(points, max_iters=5000, deadline=None, temp_start=100.0, temp_end=1.0):
    """
    Fast local search with simulated annealing features.
    Optimized for speed by reusing line map logic where possible, 
    though for N=16, full recompute is fast. The key is reducing Python overhead.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    current = [list(p) for p in points]
    current_line_map = _compute_line_map(current)
    current_score = _score_from_line_map(current_line_map)
    if _max_collinear_from_line_map(current_line_map) >= 5:
        return current, -1
        
    best = [list(p) for p in current]
    best_score = current_score
    best_line_map = current_line_map
    
    total_iters = max_iters
    for it in range(total_iters):
        if time.time() >= deadline:
            break
        
        # Annealing temperature
        progress = it / total_iters
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_point = list(current[idx])
        
        # Generate candidates
        # 1. Random nearby points
        candidates = []
        for _ in range(5):
            p = _random_point()
            if p != old_point:
                candidates.append(p)
        
        # 2. Points that complete 3-lines
        # Find lines with 3 points not including idx
        for line, idx_set in current_line_map.items():
            if len(idx_set) == 3 and idx not in idx_set:
                a, b, c = line
                # Try to find an integer point on this line within bounds
                # a*x + b*y = c
                # This is expensive to do fully, so we sample a few
                if a != 0:
                    # x = (c - b*y) / a
                    # Pick a few y values
                    for y in range(-10, 11):
                        num = c - b * y
                        if num % a == 0:
                            x = num // a
                            if -1000 <= x <= 1000:
                                p = [x, y]
                                if p != old_point:
                                    candidates.append(p)
                elif b != 0:
                    # y = (c - a*x) / b
                    for x in range(-10, 11):
                        num = c - a * x
                        if num % b == 0:
                            y = num // b
                            if -1000 <= y <= 1000:
                                p = [x, y]
                                if p != old_point:
                                    candidates.append(p)
        
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        best_candidate = None
        best_candidate_score = current_score
        best_candidate_line_map = None
        
        # Evaluate top few candidates
        for cand in candidates[:10]:
            # Check distinctness
            is_distinct = True
            for j, p in enumerate(current):
                if j != idx and p == cand:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            # Try the move
            current[idx] = cand
            new_line_map = _compute_line_map(current)
            max_col = _max_collinear_from_line_map(new_line_map)
            if max_col >= 5:
                current[idx] = old_point
                continue
                
            new_score = _score_from_line_map(new_line_map)
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = cand
                best_candidate_line_map = new_line_map
            
            # Accept worse moves with probability
            if new_score < best_candidate_score:
                delta = new_score - best_candidate_score
                if delta < 0:
                    prob = math.exp(delta / max(temp, 0.01))
                    if random.random() < prob:
                        best_candidate_score = new_score
                        best_candidate = cand
                        best_candidate_line_map = new_line_map

        if best_candidate is not None:
            current[idx] = best_candidate
            current_score = best_candidate_score
            current_line_map = best_candidate_line_map
            
            if current_score > best_score:
                best_score = current_score
                best = [list(p) for p in current]
                best_line_map = current_line_map
        else:
            current[idx] = old_point

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Phase 1: Start with 4x4 grid
    grid_points = _generate_grid_config(4, 0)
    score = _score(grid_points)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in grid_points]
    
    # Phase 2: Fast Local Search from grid
    if time.time() < deadline:
        improved, imp_score = _fast_local_search(best_points, max_iters=20000, deadline=deadline)
        if imp_score > best_score:
            best_score = imp_score
            best_points = [list(p) for p in improved]
    
    # Phase 3: Multiple restarts with diverse starts
    for restart in range(20):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            start_config = [list(p) for p in best_points]
        elif restart % 2 == 0:
            # Perturbed grid
            base = random.choice([3, 4, 5])
            start_config = _generate_grid_config(base, random.randint(-2, 2))
            while len(start_config) < 16:
                p = _random_point()
                if p not in start_config:
                    start_config.append(p)
            start_config = start_config[:16]
        else:
            # Diverse random
            start_config = _generate_diverse_config()
        
        # Run fast local search
        if time.time() < deadline:
            improved, imp_score = _fast_local_search(start_config, max_iters=5000, deadline=deadline)
            if imp_score > best_score:
                best_score = imp_score
                best_points = [list(p) for p in improved]
    
    # Final check
    if best_points is None or not _validate(best_points):
        return _generate_grid_config(4, 0)
        
    return best_points

def _validate(points):
    """Validate that points are distinct, within bounds, and no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    line_map = _compute_line_map(points)
    if _max_collinear_from_line_map(line_map) >= 5:
        return False
    return True
# EVOLVE-BLOCK-END
