import numpy as np
import time
import math
from math import gcd
import random

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) for A*x + B*y + C = 0."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of points.
    Returns (score, lines_array) where score is the sum of incidences of the top 27 lines.
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    # Step 1: Find all unique lines defined by pairs of points
    line_set = set()
    pts = points
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            # Line through (x1,y1) and (x2,y2)
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - x1 * y2
            
            key = _normalize_line(A, B, C)
            if key is not None:
                line_set.add(key)
                
    # Step 2: Count incidences for each line
    counts = []
    for (A, B, C) in line_set:
        count = 0
        for p in pts:
            x, y = p
            if A * x + B * y + C == 0:
                count += 1
        counts.append((count, (A, B, C)))
        
    # Step 3: Sort by count descending, then by line tuple for determinism
    counts.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 4: Select top 27
    if len(counts) >= 27:
        selected = counts[:27]
    else:
        selected = counts
        # Pad with dummy lines if necessary (should not happen for 54 points)
        while len(selected) < 27:
            selected.append((0, (0, 0, 1)))
            
    score = sum(c for c, _ in selected)
    lines = np.array([l for _, l in selected], dtype=np.int64)
    
    return score, lines

def _generate_grid(a, b):
    """Generate an a x b grid of points starting at (0,0)."""
    xs = np.arange(0, a)
    ys = np.arange(0, b)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _check_valid_points(points):
    """Basic validity check for points."""
    if len(points) != 54:
        return False
    if points.shape != (54, 2):
        return False
    # Check for duplicates
    pt_set = set(map(tuple, points))
    if len(pt_set) != 54:
        return False
    return True

def _local_search(points, deadline, best_score, best_points, best_lines):
    """
    Perform local search by moving individual points to nearby integer coordinates.
    Returns updated best_score, best_points, best_lines.
    """
    n = len(points)
    # Neighborhood: 8-neighbors + stay
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    
    current_points = points.copy()
    current_score, _ = _compute_top27(current_points)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            for dx, dy in neighbors:
                if time.time() > deadline:
                    return best_score, best_points, best_lines
                
                new_points = current_points.copy()
                new_points[idx] = current_points[idx] + np.array([dx, dy])
                
                if not _check_valid_points(new_points):
                    continue
                    
                new_score, new_lines = _compute_top27(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    best_lines = new_lines
                    current_points = new_points.copy()
                    current_score = new_score
                    improved = True
                    break # Restart neighbor loop for this point? No, just break to next point or restart?
                         # Breaking inner loop means we don't try other neighbors for this point in this pass.
                         # This is "first improvement".
                         # To be more thorough, we could continue to check other neighbors for this point.
                         # But that is expensive. First improvement is standard.

    return best_score, best_points, best_lines

def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    """
    Simple simulated annealing to escape local optima.
    """
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while time.time() < deadline and T > T_min:
        # Pick random point and random move
        idx = np.random.randint(n)
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        if dx == 0 and dy == 0:
            T *= cooling
            continue
        
        new_points = current.copy()
        new_points[idx] = current[idx] + np.array([dx, dy])
        
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, new_lines = _compute_top27(new_points)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                best_lines = new_lines
        
        T *= cooling
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses a combination of grid configurations, local search, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    np.random.seed(42)
    random.seed(42)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score, lines):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines

    # Phase 1: Systematic grid search with modifications
    # The 6x9 grid is the strongest candidate.
    grid_configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
    ]
    
    for a, b in grid_configs:
        if time.time() > deadline - 120:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
    # Phase 2: Local search from best grid
    if best_points is not None and time.time() < deadline - 60:
        best_score, best_points, best_lines = _local_search(
            best_points, deadline - 30, best_score, best_points, best_lines
        )
        
    # Phase 3: Simulated Annealing to explore further
    if best_points is not None and time.time() < deadline - 10:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline - 5, best_score, best_points, best_lines
        )
        
    # Fallback if something went wrong
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _compute_top27(pts)
        return pts, lines

    return best_points, best_lines
