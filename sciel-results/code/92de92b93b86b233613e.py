"""
Constructs 16 integer points maximizing lines with exactly 4 collinear points.
Strategy: Direct constructive embedding of the affine plane AG(2,4) using
binary representation mapping, followed by local search refinement.
"""

import time
import itertools
from math import gcd

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Get normalized line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)  # vertical line x = x1
    if dy == 0:
        return (0, 1, y1)  # horizontal line y = y1
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_4_lines(points):
    """Count lines containing exactly 4 of the given points, and check no 5 collinear."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    
    point_line_count = {}
    for key, pair_count in line_counts.items():
        if pair_count >= 6:  # C(4,2)=6 means at least 4 points
            # Count actual points on this line
            dx, dy, c = key
            count = 0
            for p in points:
                x, y = p
                if dy * x - dx * y == c:
                    count += 1
            if count >= 5:
                return -1  # Invalid: 5 collinear
            if count == 4:
                point_line_count[key] = 4
    return len(point_line_count)

def _evaluate(points):
    """Evaluate a configuration: returns (num_4_lines, is_valid)."""
    # Check distinctness
    if len(set(map(tuple, points))) != len(points):
        return (0, False)
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return (0, False)
    # Count 4-lines and check no 5-collinear
    n4 = _count_4_lines(points)
    if n4 < 0:
        return (0, False)
    return (n4, True)

def _ag24_embedding():
    """
    Embed AG(2,4) into integer lattice.
    AG(2,4) has 16 points and 20 lines of size 4.
    We map points (a,b) in F4^2 to integers.
    F4 = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
    Represent F4 elements as 2-bit integers: 00, 01, 11, 10 (for 0, 1, alpha+1, alpha).
    Use a non-linear mapping to avoid trivial grid structure.
    """
    # Map F4 elements to integers 0..3
    # 0 -> 0, 1 -> 1, alpha+1 -> 2, alpha -> 3
    # Points in AG(2,4) are pairs (x, y) where x, y in F4
    # We map to (X, Y) in Z^2 using a basis that preserves some structure but breaks others
    # Simple mapping: X = 4*x + y, Y = x + 2*y might create too many collinearities
    # Let's try a more scrambled mapping to break unwanted collinearities while keeping the 20 lines
    
    points = []
    for x in range(4):
        for y in range(4):
            # Use a non-linear transformation to scatter points
            # This is a heuristic mapping; we'll refine with local search
            X = x * 2 + y
            Y = x + y * 2
            points.append([X, Y])
    return points

def _try_local_search(points, deadline, max_iter=10000):
    """
    Local search to improve the number of 4-point lines.
    Moves one point at a time to a nearby position or a position that aligns with existing lines.
    """
    best_points = [p[:] for p in points]
    best_score, _ = _evaluate(best_points)
    
    if not _ or best_score <= 0:
        return best_points, best_score

    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Shuffle point order to avoid bias
        indices = list(range(16))
        import random
        random.shuffle(indices)
        
        for i in indices:
            if time.time() >= deadline:
                break
            orig = best_points[i]
            # Try moving point i to nearby positions
            # Also try positions that might align with existing 3-point lines
            candidates = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig[0] + dx
                    new_y = orig[1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    # Check if position is free
                    occupied = False
                    for j in range(16):
                        if j != i and best_points[j][0] == new_x and best_points[j][1] == new_y:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append([new_x, new_y])
            
            # Also add some random distant positions to escape local minima
            for _ in range(5):
                nx = (orig[0] + random.randint(-20, 20))
                ny = (orig[1] + random.randint(-20, 20))
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    occupied = False
                    for j in range(16):
                        if j != i and best_points[j][0] == nx and best_points[j][1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        candidates.append([nx, ny])
            
            for new_pos in candidates:
                candidate = [p[:] for p in best_points]
                candidate[i] = new_pos
                score, valid = _evaluate(candidate)
                if valid and score > best_score:
                    best_points = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return best_points, best_score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with AG(2,4) embedding
    points = _ag24_embedding()
    best_points, best_score = _try_local_search(points, deadline)
    
    # Also try a few other structured configurations as fallbacks
    if time.time() < deadline:
        # 4x4 grid
        grid = [[x, y] for x in range(4) for y in range(4)]
        p2, s2 = _try_local_search(grid, deadline)
        if s2 > best_score:
            best_points, best_score = p2, s2
    
    # Try scaled grid
    if time.time() < deadline:
        grid2 = [[x * 2, y * 2] for x in range(4) for y in range(4)]
        p3, s3 = _try_local_search(grid2, deadline)
        if s3 > best_score:
            best_points, best_score = p3, s3
    
    return best_points

# EVOLVE-BLOCK-END
