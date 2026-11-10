# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """
    points: (N, 2)
    lines: (M, 3)
    Returns number of (point, line) pairs where A*x + B*y + C == 0.
    """
    N = len(points)
    M = len(lines)
    if N == 0 or M == 0:
        return 0
    # points[:, 0] is (N,), lines[:, 0] is (M,)
    # We want a matrix of shape (M, N) or (N, M)
    # Let's do (N, M)
    # px: (N, 1), A: (1, M) -> (N, M)
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return np.sum(total == 0)

def _get_best_lines(points, num_lines=27):
    """
    Find the best num_lines lines for the given points.
    A line is defined by any pair of distinct points.
    We count how many points lie on each line.
    We select the top num_lines distinct lines.
    """
    n = len(points)
    line_counts = {}
    
    # Generate all lines from pairs of points
    for i in range(n):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
            line_counts[norm] += 1
            
    # Sort lines by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    selected = []
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        selected.append(line)
        
    # If we don't have enough lines (e.g., all points collinear or too few points), pad
    if len(selected) < num_lines:
        # Add some generic lines
        all_lines = set(line_counts.keys())
        for line in all_lines:
            if line not in selected:
                selected.append(line)
                if len(selected) >= num_lines:
                    break
        # If still not enough, generate arbitrary lines
        while len(selected) < num_lines:
            # Use small integers to avoid overflow issues, though int64 is fine
            # Just pick distinct lines
            c = len(selected)
            selected.append(_normalize_line(1, 0, -c))
            
    return np.array(selected[:num_lines], dtype=np.int64)

def _try_grid_configuration(kx, ky, deadline):
    """Try a kx by ky grid configuration."""
    if kx * ky != 54:
        return None
    
    xs = np.arange(1, kx + 1)
    ys = np.arange(1, ky + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = _get_best_lines(points, 27)
    if lines is None or len(lines) < 27:
        return None
    
    lines_arr = np.array(lines[:27], dtype=np.int64)
    inc = _incidence_count(points, lines_arr)
    return points, lines_arr, inc

def _local_search(points, lines, max_iterations=100, deadline=None):
    """Local search to improve the configuration."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = _incidence_count(best_points, best_lines)
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try moving points
        for i in range(len(best_points)):
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = best_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    
                    # Check for duplicates
                    if len(np.unique(new_points, axis=0)) != len(new_points):
                        continue
                    
                    new_lines = _get_best_lines(new_points, 27)
                    if new_lines is None or len(new_lines) < 27:
                        continue
                    
                    new_lines_arr = np.array(new_lines[:27], dtype=np.int64)
                    new_inc = _incidence_count(new_points, new_lines_arr)
                    
                    if new_inc > best_inc:
                        best_points = new_points
                        best_lines = new_lines_arr
                        best_inc = new_inc
                        improved = True
        
        # Try changing lines
        for i in range(len(best_lines)):
            # Try replacing line i with a better one
            candidate_lines = _get_best_lines(best_points, 27)
            if candidate_lines is None or len(candidate_lines) < 27:
                continue
            
            # Try swapping
            for j in range(len(candidate_lines)):
                if j == i:
                    continue
                new_lines = best_lines.copy()
                new_lines[i] = candidate_lines[j]
                
                # Check for duplicates
                norm_lines = [_normalize_line(*l) for l in new_lines]
                if len(set(norm_lines)) != len(norm_lines):
                    continue
                
                new_lines_arr = np.array(new_lines, dtype=np.int64)
                new_inc = _incidence_count(best_points, new_lines_arr)
                
                if new_inc > best_inc:
                    best_lines = new_lines_arr
                    best_inc = new_inc
                    improved = True
        
        if not improved:
            break
    
    return best_points, best_lines, best_inc

def _finite_field_construction(p=59):
    """
    Construct 54 points in F_p x F_p such that we can select 27 lines
    to maximize incidences.
    
    Strategy:
    Let P be a set of 54 points.
    Let L be a set of 27 lines.
    We want to maximize I = sum_{l in L} |P ∩ l|.
    
    In an affine plane of order q, every line has q points.
    If we could pick P such that every line in L has many points from P, we'd do well.
    
    Consider the case where P is a subset of a line? No, distinct points.
    
    Let's try a specific structure.
    Let P be the graph of a function f: X -> F_p where |X| = 54? No, p=59, so X can be size 54.
    If P is the graph of a function, any non-vertical line intersects P in at most 1 point?
    No, a line y = mx + c intersects the graph of f in points where f(x) = mx + c.
    If f is a polynomial of degree d, a line intersects it in at most d points.
    
    If we pick f(x) = x^2, lines intersect in at most 2 points.
    If we pick f(x) = x^3, lines intersect in at most 3 points.
    
    Let's try to maximize the number of intersections.
    We have 27 lines.
    If we choose the lines to be the "secants" with the most points.
    
    Actually, let's look at the dual.
    Points in dual plane correspond to lines in primal.
    Lines in dual plane correspond to points in primal.
    
    Let's stick to the primal.
    If we take P to be a set of points on a conic (e.g., parabola), any line intersects P in at most 2 points.
    Max incidences = 27 * 2 = 54.
    
    If we take P to be a set of points on a cubic curve, any line intersects P in at most 3 points.
    Max incidences = 27 * 3 = 81.
    
    Can we do better?
    What if P is not on a single curve?
    
    Let's try a grid in F_p.
    P = { (x, y) | x in X, y in Y } where |X|*|Y| = 54.
    Since 54 = 6 * 9, let X be 6 elements, Y be 9 elements.
    Lines:
    Vertical lines x = x0 for x0 in X. There are 6 such lines. Each contains 9 points.
    Horizontal lines y = y0 for y0 in Y. There are 9 such lines. Each contains 6 points.
    Total lines available: 15.
    We need 27 lines.
    We can add diagonal lines.
    
    Let's evaluate the grid 6x9 in F_59.
    X = {0, 1, 2, 3, 4, 5}
    Y = {0, 1, 2, 3, 4, 5, 6, 7, 8}
    P = X x Y.
    
    Lines:
    6 verticals, 9 horizontals.
    Incidences from these: 6*9 + 9*6 = 54 + 54 = 108.
    We have 12 more lines to choose.
    
    Diagonal lines y = m x + c.
    How many points from P does a line y = m x + c contain?
    For a fixed m, c, the number of solutions to y = m x + c with x in X, y in Y.
    This depends on the specific X, Y.
    
    If X, Y are intervals in F_p, the number of intersections is roughly |X| (if the line doesn't wrap around).
    Since p=59 is large, and X, Y are small, wrap-around is unlikely for small slopes.
    So a diagonal line might contain ~6 points.
    
    If we pick 12 diagonal lines, each with ~6 points, we add ~72 incidences.
    Total ~180.
    
    Let's implement this and see.
    We can optimize the choice of X and Y.
    """
    p = 59
    # Factors of 54: 1x54, 2x27, 3x18, 6x9
    # 6x9 is more square-ish, likely better for diagonals.
    kx, ky = 6, 9
    X = list(range(kx))
    Y = list(range(ky))
    
    points = []
    for x in X:
        for y in Y:
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    lines = _get_best_lines(points, 27)
    inc = _incidence_count(points, lines)
    
    return points, lines, inc

def _local_search_ff(points, lines, deadline):
    """
    Local search specifically for finite field configurations.
    Moves points in F_p.
    """
    p = 59
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = _incidence_count(best_points, best_lines)
    
    # Shuffle points to break symmetry?
    # random.shuffle(best_points) -- can't shuffle 2D array easily, but we can permute rows.
    
    n = len(best_points)
    
    # We will perform hill climbing.
    # At each step, pick a random point, move it to a random new location in F_p x F_p.
    # Check for duplicates.
    # Recompute best lines and score.
    
    # To speed up, we might restrict moves to nearby points?
    # In F_p, "nearby" is not well-defined.
    # But we can try small changes.
    
    # Let's try moving one point at a time.
    # Number of iterations:
    # _get_best_lines takes ~0.1ms?
    # 590 seconds / 0.1ms = 5,900,000 iterations.
    # This is a lot.
    
    # However, Python overhead is high.
    # Let's estimate 1ms per iteration.
    # 590 seconds / 1ms = 590,000 iterations.
    
    # We will run until deadline.
    
    while time.time() < deadline - 10:
        improved = False
        
        # Pick a random point to move
        i = random.randint(0, n - 1)
        old_x, old_y = int(best_points[i, 0]), int(best_points[i, 1])
        
        # Try a few random moves
        for _ in range(10):
            new_x = random.randint(0, p - 1)
            new_y = random.randint(0, p - 1)
            
            if new_x == old_x and new_y == old_y:
                continue
                
            new_points = best_points.copy()
            new_points[i, 0] = new_x
            new_points[i, 1] = new_y
            
            # Check for duplicates
            if len(np.unique(new_points, axis=0)) != n:
                continue
            
            new_lines = _get_best_lines(new_points, 27)
            new_inc = _incidence_count(new_points, new_lines)
            
            if new_inc > best_inc:
                best_points = new_points
                best_lines = new_lines
                best_inc = new_inc
                improved = True
                break # Accept and move to next point
        
        if not improved:
            # Try a more aggressive move? Or restart?
            # Let's just continue.
            pass
            
    return best_points, best_lines, best_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Try grid configurations in integers
    grid_configs = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    for kx, ky in grid_configs:
        if time.time() > deadline:
            break
        result = _try_grid_configuration(kx, ky, deadline)
        if result is not None:
            points, lines, inc = result
            if inc > best_inc:
                best_points = points
                best_lines = lines
                best_inc = inc
    
    # Try finite field construction
    if time.time() < deadline - 60:
        ff_points, ff_lines, ff_inc = _finite_field_construction()
        if ff_inc > best_inc:
            best_points = ff_points
            best_lines = ff_lines
            best_inc = ff_inc
            
    # If we have a result, try local search
    if best_points is not None:
        if time.time() < deadline - 30:
            # Determine if we are in FF or Integer space
            # If max coordinate < 59, assume FF
            max_coord = np.max(best_points)
            if max_coord < 59:
                best_points, best_lines, best_inc = _local_search_ff(
                    best_points, best_lines, deadline
                )
            else:
                best_points, best_lines, best_inc = _local_search(
                    best_points, best_lines, max_iterations=50, deadline=deadline
                )
    
    # Fallback: simple grid
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = _get_best_lines(best_points, 27)
        if best_lines is None or len(best_lines) < 27:
            # Create default lines
            best_lines = []
            for c in xs:
                best_lines.append((1, 0, -c))
            for c in ys:
                best_lines.append((0, 1, -c))
            while len(best_lines) < 27:
                best_lines.append((1, 1, 0))
        best_lines = np.array(best_lines[:27], dtype=np.int64)
    
    # Verify the result
    final_incidents = _incidence_count(best_points, best_lines)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
