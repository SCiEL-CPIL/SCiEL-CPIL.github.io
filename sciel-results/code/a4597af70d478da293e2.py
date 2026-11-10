# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from fractions import Fraction


def _time_left(deadline):
    """Check if we have time left before the deadline."""
    return time.time() < deadline


def _canon_line(A, B, C):
    """Canonical form of a line (A, B, C) -> (a, b, c) where gcd(a,b,c)=1 and first nonzero is positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Get line (A, B, C) through two integer points, in canonical form."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
    A = dy
    B = -dx
    C = x1 * y2 - x2 * y1
    return _canon_line(A, B, C)


def _incidence_count(points, lines):
    """Count point-line incidences exactly using int64 arithmetic."""
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


def _is_valid(points, lines):
    """Check if points and lines are valid per the task contract."""
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN/inf and integrality
    for arr in [points, lines]:
        if not np.all(np.isfinite(arr.astype(float))):
            return False
        if not np.allclose(arr, np.round(arr), atol=1e-6):
            return False
    
    # Convert to int64
    points_int = np.round(points).astype(np.int64)
    lines_int = np.round(lines).astype(np.int64)
    
    # Check for repeated points
    if len(set(map(tuple, points_int))) != 54:
        return False
    
    # Check lines
    seen_lines = set()
    for i in range(27):
        A, B, C = lines_int[i]
        if A == 0 and B == 0:
            return False
        canon = _canon_line(A, B, C)
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    
    return True


def _grid_construction():
    """Classic Szemeredi-Trotter grid construction with k=3."""
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


def _star_construction():
    """Star construction: points on a circle, lines through pairs."""
    # Place 54 points on a circle of radius R
    R = 100
    angles = np.linspace(0, 2 * np.pi, 54, endpoint=False)
    points = np.column_stack([
        np.round(R * np.cos(angles)).astype(np.int64),
        np.round(R * np.sin(angles)).astype(np.int64)
    ])
    
    # Remove duplicate points
    unique_points = []
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we have fewer than 54, add more points
    while len(unique_points) < 54:
        x = random.randint(-R, R)
        y = random.randint(-R, R)
        if (x, y) not in seen:
            seen.add((x, y))
            unique_points.append((x, y))
    
    points = np.array(unique_points[:54], dtype=np.int64)
    
    # Select 27 lines from all possible lines through pairs
    all_lines = set()
    for i in range(54):
        for j in range(i + 1, 54):
            line = _line_from_two_points(points[i], points[j])
            if line is not None:
                all_lines.add(line)
    
    all_lines = list(all_lines)
    random.shuffle(all_lines)
    lines = np.array(all_lines[:27], dtype=np.int64)
    
    return points, lines


def _random_construction():
    """Random construction with some structure."""
    # Generate random points in a bounded region
    points = []
    seen = set()
    while len(points) < 54:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    points = np.array(points, dtype=np.int64)
    
    # Generate lines from point pairs
    all_lines = set()
    for i in range(54):
        for j in range(i + 1, 54):
            line = _line_from_two_points(points[i], points[j])
            if line is not None:
                all_lines.add(line)
    
    all_lines = list(all_lines)
    random.shuffle(all_lines)
    lines = np.array(all_lines[:27], dtype=np.int64)
    
    return points, lines


def _local_search(points, lines, deadline, max_iter=10000):
    """Local search to improve incidences by moving points or lines."""
    points = points.copy()
    lines = lines.copy()
    current_score = _incidence_count(points, lines)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    for iteration in range(max_iter):
        if not _time_left(deadline):
            break
        
        # Try moving a random point
        if random.random() < 0.5:
            idx = random.randint(0, 53)
            old_x, old_y = points[idx]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    # Check if point already exists
                    if any(points[i][0] == new_x and points[i][1] == new_y for i in range(54) if i != idx):
                        continue
                    
                    points[idx] = [new_x, new_y]
                    new_score = _incidence_count(points, lines)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_points = points.copy()
                            best_lines = lines.copy()
                        break
                    else:
                        points[idx] = [old_x, old_y]
                else:
                    continue
                break
        else:
            # Try replacing a line with a better one
            idx = random.randint(0, 26)
            old_line = lines[idx].copy()
            
            # Find a line that goes through at least 3 points
            best_new_line = None
            best_new_score = current_score
            
            for i in range(54):
                for j in range(i + 1, 54):
                    for k in range(j + 1, 54):
                        # Check if these 3 points are collinear
                        p1, p2, p3 = points[i], points[j], points[k]
                        area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                        if area == 0:
                            line = _line_from_two_points(p1, p2)
                            if line is not None:
                                # Check if this line is already in use
                                if any(_canon_line(*lines[l]) == line for l in range(27) if l != idx):
                                    continue
                                lines[idx] = line
                                new_score = _incidence_count(points, lines)
                                if new_score > best_new_score:
                                    best_new_score = new_score
                                    best_new_line = line
                                lines[idx] = old_line
                        if not _time_left(deadline):
                            break
                    if not _time_left(deadline):
                        break
                if not _time_left(deadline):
                    break
            
            if best_new_line is not None and best_new_score > current_score:
                lines[idx] = best_new_line
                current_score = best_new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
    
    return best_points, best_lines, best_score


def _evaluate(points, lines):
    """Evaluate a configuration, returning (valid, score)."""
    if not _is_valid(points, lines):
        return False, 0
    score = _incidence_count(points, lines)
    return True, score


def _solve_one(deadline, seed=0):
    """Solve one instance with a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Try multiple initial constructions
    constructions = [
        _grid_construction,
        _star_construction,
        _random_construction
    ]
    
    best_points = None
    best_lines = None
    best_score = 0
    
    for i, construct in enumerate(constructions):
        if not _time_left(deadline):
            break
        
        points, lines = construct()
        valid, score = _evaluate(points, lines)
        
        if valid and score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
        
        # Local search from this starting point
        if _time_left(deadline):
            points_ls, lines_ls, score_ls = _local_search(points, lines, deadline, max_iter=5000)
            if score_ls > best_score:
                best_score = score_ls
                best_points = points_ls
                best_lines = lines_ls
    
    return best_points, best_lines, best_score


def _run_search(deadline):
    """Run multiple searches with different seeds."""
    best_points = None
    best_lines = None
    best_score = 0
    
    for seed in range(20):
        if not _time_left(deadline):
            break
        
        points, lines, score = _solve_one(deadline, seed=seed)
        
        if points is not None and score > best_score:
            best_score = score
            best_points = points
            best_lines = lines
    
    return best_points, best_lines, best_score


def _finalize(points, lines):
    """Final validation and cleanup."""
    # Round to int64
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    # Ensure shapes
    if points.shape != (54, 2):
        raise ValueError("Invalid points shape")
    if lines.shape != (27, 3):
        raise ValueError("Invalid lines shape")
    
    return points, lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.

    Uses multiple diverse initial constructions and local search to find a high-incidence
    configuration. Includes a deadline guard to ensure we return before timeout.

    Returns:
        points: np.ndarray of shape (54, 2), integer (x, y) lattice coordinates.
        lines : np.ndarray of shape (27, 3), integer (A, B, C) with A*x + B*y + C = 0.
    """
    # Set deadline with safety margin (630s limit, use 600s to be safe)
    deadline = time.time() + 600
    
    # Run search
    best_points, best_lines, best_score = _run_search(deadline)
    
    # If search failed, fall back to grid construction
    if best_points is None:
        best_points, best_lines = _grid_construction()
    
    # Finalize
    points, lines = _finalize(best_points, best_lines)
    
    return points, lines


# EVOLVE-BLOCK-END
