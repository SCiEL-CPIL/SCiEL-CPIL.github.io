# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _canon_line(A, B, C):
    """Canonical form: divide by gcd, make first nonzero positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly using int64 arithmetic."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _count_incidences_fast(points, lines):
    """Vectorized incidence count."""
    # points: (n, 2), lines: (m, 3)
    # Compute A*x + B*y + C for all pairs
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    # For each line, check all points
    total = 0
    for j in range(len(lines)):
        val = A[j] * px + B[j] * py + C[j]
        total += np.sum(val == 0)
    return int(total)

def _valid_points(points):
    """Check points are valid: unique, integer, within bounds."""
    if len(points) != 54:
        return False
    seen = set()
    for p in points:
        x, y = int(p[0]), int(p[1])
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _valid_lines(lines):
    """Check lines are valid: no degenerate, no duplicates after canonicalization."""
    if len(lines) != 27:
        return False
    seen = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        c = _canon_line(A, B, C)
        if c in seen:
            return False
        seen.add(c)
    return True

def _is_valid(points, lines):
    return _valid_points(points) and _valid_lines(lines)

def _initial_grid_config(x_range, y_range, line_pool):
    """
    Generate an initial configuration from a grid.
    x_range, y_range: ranges for grid coordinates
    line_pool: list of candidate lines (A, B, C)
    """
    xs = list(x_range)
    ys = list(y_range)
    grid_points = [(x, y) for x in xs for y in ys]
    
    if len(grid_points) < 54:
        return None, None
    
    # Select 54 points from grid
    random.shuffle(grid_points)
    points = np.array(grid_points[:54], dtype=np.int64)
    
    # Select 27 lines from pool
    random.shuffle(line_pool)
    lines = np.array(line_pool[:27], dtype=np.int64)
    
    return points, lines

def _local_search(points, lines, deadline, max_iter=5000):
    """
    Local search: try moving individual points or swapping lines.
    Returns improved (points, lines, count).
    """
    count = _count_incidences_fast(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_count = count
    
    n = len(points)
    m = len(lines)
    
    # Precompute for speed
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try moving each point
        for i in range(n):
            if time.time() > deadline:
                break
            x, y = int(points[i, 0]), int(points[i, 1])
            
            # Try nearby positions
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    new_points = points.copy()
                    new_points[i] = [nx, ny]
                    
                    if not _valid_points(new_points):
                        continue
                    
                    new_count = _count_incidences_fast(new_points, lines)
                    if new_count > best_count:
                        best_count = new_count
                        best_points = new_points
                        improved = True
                        # Continue searching from new position
                        points = new_points
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swapping lines
        for i in range(m):
            if time.time() > deadline:
                break
            for j in range(i + 1, m):
                new_lines = lines.copy()
                new_lines[i], new_lines[j] = lines[j], lines[i]
                # This doesn't change anything since we're just swapping
                # Instead, try replacing a line with a better one
                break
        
        # Try replacing individual lines
        for j in range(m):
            if time.time() > deadline:
                break
            # Generate candidate lines through pairs of points
            # Pick random pairs of points and create lines through them
            for _ in range(10):
                i1, i2 = random.sample(range(n), 2)
                x1, y1 = int(points[i1, 0]), int(points[i1, 1])
                x2, y2 = int(points[i2, 0]), int(points[i2, 1])
                
                # Line through (x1,y1) and (x2,y2)
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                
                new_lines = lines.copy()
                new_lines[j] = [A, B, C]
                
                if not _valid_lines(new_lines):
                    continue
                
                new_count = _count_incidences_fast(points, new_lines)
                if new_count > best_count:
                    best_count = new_count
                    best_lines = new_lines
                    improved = True
                    lines = new_lines
                    break
            if improved:
                break
        
        if not improved:
            # Try more aggressive moves: move multiple points
            for _ in range(5):
                if time.time() > deadline:
                    break
                # Move 2 random points
                idxs = random.sample(range(n), 2)
                new_points = points.copy()
                for idx in idxs:
                    x, y = int(new_points[idx, 0]), int(new_points[idx, 1])
                    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
                    new_points[idx] = [x + dx, y + dy]
                
                if not _valid_points(new_points):
                    continue
                
                new_count = _count_incidences_fast(new_points, lines)
                if new_count > best_count:
                    best_count = new_count
                    best_points = new_points
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_points, best_lines, best_count

def _generate_line_pool(points, num_lines=100):
    """Generate a pool of candidate lines based on point pairs."""
    lines = []
    n = len(points)
    seen = set()
    
    # Lines through pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
            
            c = _canon_line(A, B, C)
            if c not in seen:
                seen.add(c)
                lines.append(c)
    
    # Add some random lines
    for _ in range(200):
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        if A == 0 and B == 0:
            continue
        C = random.randint(-20, 20)
        c = _canon_line(A, B, C)
        if c not in seen:
            seen.add(c)
            lines.append(c)
    
    return lines

def _grid_search(deadline):
    """
    Try various grid configurations and optimize.
    """
    best_points = None
    best_lines = None
    best_count = 0
    
    # Try different grid sizes
    grid_configs = [
        (range(1, 8), range(1, 8)),    # 7x7 = 49, need to add more
        (range(1, 9), range(1, 7)),    # 8x6 = 48
        (range(1, 9), range(1, 8)),    # 8x7 = 56
        (range(1, 8), range(1, 9)),    # 7x8 = 56
        (range(1, 10), range(1, 6)),   # 9x5 = 45
        (range(1, 6), range(1, 10)),   # 5x9 = 45
        (range(1, 12), range(1, 5)),   # 11x4 = 44
        (range(1, 5), range(1, 12)),   # 4x11 = 44
        (range(1, 9), range(1, 9)),    # 8x8 = 64
        (range(1, 6), range(1, 9)),    # 5x8 = 40
        (range(1, 9), range(1, 6)),    # 9x5 = 45
    ]
    
    for x_range, y_range in grid_configs:
        if time.time() > deadline:
            break
        
        xs = list(x_range)
        ys = list(y_range)
        grid_points = [(x, y) for x in xs for y in ys]
        
        if len(grid_points) < 54:
            continue
        
        # Try multiple random selections
        for trial in range(5):
            if time.time() > deadline:
                break
            
            random.shuffle(grid_points)
            points = np.array(grid_points[:54], dtype=np.int64)
            
            # Generate line pool
            line_pool = _generate_line_pool(points, 100)
            
            if len(line_pool) < 27:
                continue
            
            random.shuffle(line_pool)
            lines = np.array(line_pool[:27], dtype=np.int64)
            
            if not _is_valid(points, lines):
                continue
            
            # Local search
            p, l, c = _local_search(points, lines, deadline, max_iter=2000)
            
            if c > best_count:
                best_count = c
                best_points = p
                best_lines = l
    
    return best_points, best_lines, best_count

def _szemeredi_baseline():
    """Classical Szemeredi-Trotter construction."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Returns (points, lines) maximizing incidences.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s to be safe
    
    # Start with baseline
    base_points, base_lines = _szemeredi_baseline()
    best_points = base_points.copy()
    best_lines = base_lines.copy()
    best_count = _count_incidences_fast(best_points, best_lines)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Grid search with local optimization
    p, l, c = _grid_search(deadline)
    if p is not None and c > best_count:
        best_points = p
        best_lines = l
        best_count = c
    
    # Additional local search from best found
    if time.time() < deadline - 60:
        p, l, c = _local_search(best_points, best_lines, deadline, max_iter=10000)
        if c > best_count:
            best_points = p
            best_lines = l
            best_count = c
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        # Fall back to baseline
        best_points, best_lines = _szemeredi_baseline()
        best_count = _count_incidences_fast(best_points, best_lines)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
