"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        g = 1
    dx_n = dx // g
    dy_n = dy // g
    # Canonical direction: first non-zero component positive
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * p1[0] - dx_n * p1[1]
    return (dx_n, dy_n, c)

def _get_line_counts(points):
    """For a set of points, return dict: line_key -> count of points on that line."""
    line_counts = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            lk = _line_key(points[i], points[j])
            if lk is None:
                continue
            # Only count if at least 2 points, but we'll filter later
            line_counts[lk] = line_counts.get(lk, 0) + 1
    # This overcounts; each pair adds 1, but we want unique points per line.
    # Better approach: for each line key, count unique points.
    # Let's redo this properly.
    line_to_points = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            lk = _line_key(points[i], points[j])
            if lk is None:
                continue
            if lk not in line_to_points:
                line_to_points[lk] = set()
            line_to_points[lk].add(i)
            line_to_points[lk].add(j)
    
    # Convert to counts, only keep lines with >= 3 points
    result = {}
    for lk, pts in line_to_points.items():
        if len(pts) >= 3:
            result[lk] = len(pts)
    return result

def _has_four_collinear(points):
    """Check if any 4 points are collinear."""
    line_counts = _get_line_counts(points)
    for count in line_counts.values():
        if count >= 4:
            return True
    return False

def _count_collinear_triples(points):
    """Count the number of 3-collinear triples in the point set."""
    line_counts = _get_line_counts(points)
    total = 0
    for count in line_counts.values():
        if count >= 3:
            total += count * (count - 1) * (count - 2) // 6
    return total

def _largest_no_three_collinear(points):
    """Find the largest subset with no three collinear using branch and bound."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute which triples are collinear
    collinear_triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                lk = _line_key(points[i], points[j])
                if lk is None:
                    continue
                # Check if k is on the same line
                dx = lk[0]
                dy = lk[1]
                c = lk[2]
                if dy * points[k][0] - dx * points[k][1] == c:
                    collinear_triples.add((i, j, k))
    
    best_size = 0
    best_subset = set()
    
    def branch(subset, candidates):
        nonlocal best_size, best_subset
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset.copy()
        
        if not candidates:
            return
        
        # Upper bound: current size + remaining candidates
        if len(subset) + len(candidates) <= best_size:
            return
        
        # Try adding each candidate
        for i, idx in enumerate(candidates):
            # Check if adding idx creates a collinear triple
            valid = True
            for j in subset:
                for k in subset:
                    if j == k:
                        continue
                    # Check if (idx, j, k) is collinear
                    if idx > j > k:
                        triple = (k, j, idx)
                    elif idx > k > j:
                        triple = (j, k, idx)
                    elif j > idx > k:
                        triple = (k, idx, j)
                    elif k > idx > j:
                        triple = (j, idx, k)
                    elif j > k > idx:
                        triple = (idx, k, j)
                    else:
                        triple = (idx, j, k)
                    if triple in collinear_triples:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_subset = subset | {idx}
                new_candidates = candidates[i+1:]
                branch(new_subset, new_candidates)
    
    branch(set(), list(range(n)))
    return best_size

def _score_config(points):
    """Score a configuration: must have no 4 collinear, then minimize largest 3-free subset."""
    if len(points) != 20:
        return float('inf')
    if _has_four_collinear(points):
        return float('inf')
    return _largest_no_three_collinear(points)

def _generate_candidate():
    """Generate a random candidate configuration."""
    points = []
    seen = set()
    while len(points) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = float('inf')
    best_points = None
    
    # Strategy: Start with structured configurations that encourage collinearity
    # then use local search to minimize the largest 3-free subset
    
    # Initial structured candidates
    candidates = []
    
    # 1. Grid-based: 4x5 grid with perturbations
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            pts = []
            for i in range(4):
                for j in range(5):
                    x = i * 10 + dx + random.randint(-2, 2)
                    y = j * 10 + dy + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        pts.append((x, y))
            if len(pts) >= 20:
                pts = pts[:20]
                if not _has_four_collinear(pts):
                    candidates.append(pts)
    
    # 2. Parabola-based: y = ax^2 + bx + c
    for a in [1, -1, 2, -2]:
        for b in range(-5, 6, 2):
            for c in range(-5, 6, 2):
                pts = []
                for x in range(-10, 10):
                    y = a * x * x + b * x + c
                    if -100 <= y <= 100:
                        pts.append((x, y))
                if len(pts) >= 20:
                    pts = pts[:20]
                    if not _has_four_collinear(pts):
                        candidates.append(pts)
    
    # 3. Random candidates
    for _ in range(50):
        pts = _generate_candidate()
        if not _has_four_collinear(pts):
            candidates.append(pts)
    
    # Evaluate initial candidates
    for pts in candidates:
        score = _score_config(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
    
    # Local search: swap points to reduce score
    current_points = best_points[:] if best_points else _generate_candidate()
    
    while time.time() < deadline:
        # Try random swaps
        improved = False
        for _ in range(20):
            i = random.randint(0, 19)
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            if (new_x, new_y) in current_points:
                continue
            
            old_pt = current_points[i]
            current_points[i] = (new_x, new_y)
            
            if _has_four_collinear(current_points):
                current_points[i] = old_pt
                continue
            
            new_score = _score_config(current_points)
            if new_score < best_score:
                best_score = new_score
                best_points = current_points[:]
                improved = True
            else:
                current_points[i] = old_pt
        
        if not improved and time.time() > start_time + 60:
            break
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
