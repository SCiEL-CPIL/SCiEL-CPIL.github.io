# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations
from math import gcd


def _normalize_line(A, B, C):
    """Normalize (A, B, C) so that gcd(|A|,|B|,|C|)=1 and first nonzero entry is positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
    # Make first nonzero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count (point, line) pairs where A*x + B*y + C == 0."""
    count = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count


def _generate_grid(w, h, offset_x=0, offset_y=0):
    """Generate a w x h grid of points."""
    xs = np.arange(w, dtype=np.int64) + offset_x
    ys = np.arange(h, dtype=np.int64) + offset_y
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _get_all_lines_for_points(points):
    """
    For a set of points, enumerate all lines passing through at least 2 points.
    Returns a dict: normalized line key -> set of point indices on that line.
    """
    pts = points.tolist()
    n = len(pts)
    line_to_indices = {}

    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in line_to_indices:
                line_to_indices[key] = set()
            line_to_indices[key].add(i)
            line_to_indices[key].add(j)

    # Keep only lines with at least 2 points
    return {k: v for k, v in line_to_indices.items() if len(v) >= 2}


def _evaluate_configuration(points, top_k_lines):
    """
    Given points and a list of top-k lines (as normalized keys), 
    compute the total incidences.
    """
    # For each line, count how many points are on it
    total = 0
    for line_key in top_k_lines:
        A, B, C = line_key
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        total += cnt
    return total


def _try_local_improvement(points, lines, deadline):
    """
    Try to improve the configuration by swapping one point for another.
    Returns improved (points, lines) or the original if no improvement found.
    """
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidences(best_points, best_lines)

    # Try moving each point to nearby integer positions
    for i in range(len(best_points)):
        if time.time() > deadline - 5:
            break
        x0, y0 = best_points[i]
        original = (x0, y0)
        
        # Try positions in a window around (x0, y0)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                # Check if this new point is already in the set
                new_point = (nx, ny)
                if new_point in set(map(tuple, best_points.tolist())):
                    continue
                
                # Tentatively replace
                test_points = best_points.copy()
                test_points[i] = [nx, ny]
                
                # Recompute incidences
                score = _count_incidences(test_points, best_lines)
                if score > best_score:
                    best_points = test_points
                    best_score = score

    # After point adjustments, try to re-select the best 27 lines
    line_dict = _get_all_lines_for_points(best_points)
    candidate_lines = sorted(line_dict.items(), key=lambda t: -len(t[1]))
    top_27 = [key for key, _ in candidate_lines[:27]]
    
    final_score = _evaluate_configuration(best_points, top_27)
    if final_score > best_score:
        best_lines = np.array(top_27, dtype=np.int64)
        best_score = final_score

    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    """
    Point-line incidence maximization with n=54 points and m=27 lines.
    Uses grid-based construction with local search improvement.
    """
    start = time.time()
    deadline = start + 620.0

    # Try several grid dimensions
    best_points = None
    best_lines = None
    best_score = -1

    # Grid dimensions that give 54 points: 1x54, 2x27, 3x18, 6x9, 9x6, 18x3, 27x2, 54x1
    # Also try some non-rectangular or shifted grids
    grid_configs = [
        (9, 6, 0, 0),
        (6, 9, 0, 0),
        (18, 3, 0, 0),
        (3, 18, 0, 0),
        (27, 2, 0, 0),
        (2, 27, 0, 0),
        # Shifted variants
        (9, 6, -4, -3),
        (6, 9, -3, -4),
        (18, 3, -9, -1),
        (3, 18, -1, -9),
    ]

    for w, h, ox, oy in grid_configs:
        if time.time() > deadline - 30:
            break

        points = _generate_grid(w, h, ox, oy)

        # Get all lines through at least 2 points
        line_dict = _get_all_lines_for_points(points)
        
        if len(line_dict) < 27:
            continue

        # Sort by number of points on the line, descending
        candidate_lines = sorted(line_dict.items(), key=lambda t: (-len(t[1]), t[0]))
        
        # Take top 27 lines
        top_27 = [key for key, _ in candidate_lines[:27]]
        lines = np.array(top_27, dtype=np.int64)

        # Evaluate
        score = _evaluate_configuration(points, top_27)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()

    # Now try local improvement on the best configuration
    if best_points is not None and time.time() < deadline - 10:
        best_points, best_lines, best_score = _try_local_improvement(
            best_points, best_lines, deadline
        )

    # Final verification
    assert best_points is not None
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)

    # Check for duplicate points
    unique_points = set(map(tuple, best_points.tolist()))
    assert len(unique_points) == 54, "Duplicate points found"

    # Check for invalid lines (A=B=0)
    for A, B, C in best_lines:
        assert not (A == 0 and B == 0), "Line with A=B=0"

    # Check for duplicate lines
    line_keys = set()
    for A, B, C in best_lines:
        key = _normalize_line(int(A), int(B), int(C))
        assert key not in line_keys, "Duplicate line found"
        line_keys.add(key)

    # Check all entries are integers
    assert np.all(np.isfinite(best_points))
    assert np.all(np.isfinite(best_lines))
    assert np.allclose(best_points, np.round(best_points))
    assert np.allclose(best_lines, np.round(best_lines))

    # Round to int64
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)

    # Final incidence count
    final_I = _count_incidences(best_points, best_lines)

    return best_points, best_lines


# EVOLVE-BLOCK-END
