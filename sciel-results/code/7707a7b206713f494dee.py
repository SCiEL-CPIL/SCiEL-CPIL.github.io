"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triples += 1
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                    continue
                for l in range(k+1, n):
                    x4, y4 = points[l]
                    if (x2-x1)*(y4-y1) == (y2-y1)*(x4-x1):
                        return True
    return False

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    best_set = [set()]
    
    # Precompute collinearity: collinear[i][j][k] = True if i,j,k are collinear
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
                best_set[0] = set(current)
            return
        
        # Pruning: even if we take all remaining, can we beat best?
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try excluding idx first (to find smaller subsets quickly)
        dfs(idx + 1, current, current_size)
        
        # Try including idx
        can_add = True
        for i, ci in enumerate(current):
            for j in range(i+1, len(current)):
                cj = current[j]
                if collinear[idx][ci][cj]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
    
    dfs(0, [], 0)
    return best[0]

def _try_line_packing(seed):
    """
    Construct 20 points by explicitly trying to create 3-point lines.
    Strategy: place points on a set of lines, ensuring no 4 are collinear.
    """
    rng = random.Random(seed)
    
    # We'll try to create a configuration with many 3-point lines.
    # Start with a base structure and add points to create lines.
    
    # Use a grid-like base but with careful placement
    # Try different base structures
    
    structures = [
        # Structure 1: Two parallel lines with 5 points each, plus 10 more
        lambda: _structure_two_lines(rng),
        # Structure 2: Star pattern
        lambda: _structure_star(rng),
        # Structure 3: Grid with perturbations optimized for 3-lines
        lambda: _structure_optimized_grid(rng),
    ]
    
    best_points = None
    best_triples = -1
    
    for struct_fn in structures:
        points = struct_fn()
        if points is None:
            continue
        
        # Validate
        if len(set(tuple(p) for p in points)) != 20:
            continue
        if _has_four_collinear(points):
            continue
        
        triples = _count_collinear_triples(points)
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in points]
    
    return best_points

def _structure_two_lines(rng):
    """Create two parallel lines with 5 points each, then add 10 more to create 3-lines."""
    # Two horizontal lines at y=0 and y=20, with 5 points each
    points = []
    for i in range(5):
        points.append([i * 25, 0])
        points.append([i * 25 + 5, 20])
    
    # Now add 10 more points to create 3-point lines
    # Try to create diagonal 3-point lines
    # We need to place points such that they form 3-point lines with existing points
    
    # Available positions: avoid collinear with 4 points
    used = set(tuple(p) for p in points)
    
    # Try to add points that create 3-point lines
    candidates = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            if (x, y) not in used:
                candidates.append((x, y))
    
    # Shuffle and try to add points that maximize 3-lines
    rng.shuffle(candidates)
    
    for cand in candidates:
        if len(points) >= 20:
            break
        new_points = points + [list(cand)]
        if len(set(tuple(p) for p in new_points)) != len(new_points):
            continue
        if _has_four_collinear(new_points):
            continue
        points.append(list(cand))
    
    if len(points) != 20:
        return None
    
    return points

def _structure_star(rng):
    """Star pattern: points arranged around a center with some on lines through center."""
    points = []
    
    # Center point
    points.append([0, 0])
    
    # Points on lines through center (but not 4 collinear)
    # Use different slopes
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for angle in angles:
        rad = angle * 3.14159265358979 / 180.0
        # Place 1-2 points on this line
        for dist in [15, 30]:
            x = int(dist * 0.866 if angle % 180 != 0 else dist)  # cos approx
            y = int(dist * 0.5 if angle % 180 != 0 else 0)  # sin approx
            # More accurate
            import math
            x = int(dist * math.cos(rad))
            y = int(dist * math.sin(rad))
            if -100 <= x <= 100 and -100 <= y <= 100:
                if (x, y) not in set(tuple(p) for p in points):
                    points.append([x, y])
                    if len(points) >= 20:
                        break
        if len(points) >= 20:
            break
    
    if len(points) != 20:
        return None
    
    # Check validity
    if _has_four_collinear(points):
        return None
    
    return points

def _structure_optimized_grid(rng):
    """Start with a grid and optimize placement to maximize 3-point lines."""
    # Base 5x4 grid
    points = []
    for row in range(4):
        for col in range(5):
            x = col * 22
            y = row * 22
            # Add small random perturbation
            x += rng.randint(-3, 3)
            y += rng.randint(-3, 3)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            points.append([x, y])
    
    # Remove duplicates
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    while len(unique) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    
    points = unique[:20]
    
    if _has_four_collinear(points):
        return None
    
    return points

def _local_search_line_aware(points, deadline):
    """
    Local search that specifically tries to create 3-point lines.
    For each point, try moving it to a position that creates a new 3-point line
    with two existing points, while not creating a 4-point line.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    
    # Precompute all pairs and their line equations
    def get_line(p1, p2):
        """Return (A, B, C) for line Ax + By + C = 0"""
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        # Normalize
        from math import gcd
        g = gcd(abs(A), gcd(abs(B), abs(C))) if A != 0 or B != 0 else 1
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def point_on_line(p, line):
        A, B, C = line
        return A * p[0] + B * p[1] + C == 0
    
    while time.time() < deadline:
        improved = False
        
        for i in range(20):
            if time.time() > deadline:
                break
            
            orig = points[i][:]
            
            # Try to move point i to create a 3-point line with two other points
            for j in range(20):
                if j == i:
                    continue
                for k in range(j + 1, 20):
                    if k == i:
                        continue
                    
                    # Line through j and k
                    line_jk = get_line(points[j], points[k])
                    
                    # Check if point i is already on this line
                    if point_on_line(orig, line_jk):
                        continue
                    
                    # Try to move point i to a position on line_jk that:
                    # 1. Is within bounds
                    # 2. Doesn't create 4-collinear
                    # 3. Creates a new 3-point line
                    
                    # Find integer points on line_jk
                    A, B, C = line_jk
                    
                    # Generate candidate positions on this line
                    candidates_on_line = []
                    if B != 0:
                        # y = (-A*x - C) / B
                        for x in range(-100, 101):
                            num = -A * x - C
                            if num % B == 0:
                                y = num // B
                                if -100 <= y <= 100:
                                    candidates_on_line.append([x, y])
                    elif A != 0:
                        # x = (-B*y - C) / A
                        for y in range(-100, 101):
                            num = -B * y - C
                            if num % A == 0:
                                x = num // A
                                if -100 <= x <= 100:
                                    candidates_on_line.append([x, y])
                    
                    # Shuffle candidates
                    rng = random.Random(hash((i, j, k, time.time())))
                    rng.shuffle(candidates_on_line)
                    
                    for new_pos in candidates_on_line[:20]:  # Try up to 20 candidates
                        # Check if this position is already used
                        if new_pos in [points[m] for m in range(20) if m != i]:
                            continue
                        
                        # Try moving point i to new_pos
                        old_pos = points[i][:]
                        points[i] = new_pos[:]
                        
                        # Check validity
                        if len(set(tuple(p) for p in points)) != 20:
                            points[i] = old_pos
                            continue
                        if _has_four_collinear(points):
                            points[i] = old_pos
                            continue
                        
                        new_triples = _count_collinear_triples(points)
                        if new_triples > current_triples:
                            current_triples = new_triples
                            improved = True
                            break
                        else:
                            points[i] = old_pos
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    best_points = None
    best_triples = -1
    
    # Try line-packing constructions
    for seed in range(50):
        if time.time() > deadline - 40:
            break
        
        points = _try_line_packing(seed)
        if points is None:
            continue
        
        triples = _count_collinear_triples(points)
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in points]
    
    # Local search on the best candidate
    if best_points is not None and time.time() < deadline - 20:
        improved = _local_search_line_aware(best_points, deadline - 10)
        if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
            improved_triples = _count_collinear_triples(improved)
            if improved_triples > best_triples:
                best_triples = improved_triples
                best_points = [p[:] for p in improved]
    
    # If we found good candidates, evaluate the best one's exact score
    if best_points is not None and time.time() < deadline - 5:
        score = _max_3free_subset(best_points, deadline - 2)
    
    # Return the best points found
    if best_points is not None:
        return best_points
    else:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
