# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _canonical_line(A, B, C):
    """Return canonical form of line (A,B,C): divide by gcd, first nonzero positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count point-line incidences: pairs (p, l) with A*x + B*y + C == 0."""
    # points: (n, 2) int64, lines: (m, 3) int64
    # For each line, check which points lie on it
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        # A*x + B*y + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += np.count_nonzero(vals == 0)
    return count

def _all_lines_through_pairs(points):
    """Generate all lines determined by pairs of points, with their incidence counts."""
    n = len(points)
    line_to_points = {}  # canonical line -> set of point indices
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y1-y2)*x + (x2-x1)*y + (x1*y2 - x2*y1) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            key = _canonical_line(A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    return line_to_points

def _top_lines(points, m):
    """Find the top m lines by incidence count."""
    line_to_points = _all_lines_through_pairs(points)
    # Sort by number of points on the line
    sorted_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
    top = []
    seen = set()
    for key, pts in sorted_lines:
        if len(top) >= m:
            break
        if key in seen:
            continue
        seen.add(key)
        top.append((key, len(pts)))
    return top

def _score_configuration(points, m=27):
    """Score = total incidences of top m lines."""
    top = _top_lines(points, m)
    return sum(cnt for _, cnt in top)

def _random_point_in_range(min_coord, max_coord, existing_points, rng):
    """Generate a random point not in existing_points within range."""
    while True:
        x = rng.randint(min_coord, max_coord)
        y = rng.randint(min_coord, max_coord)
        p = (x, y)
        if p not in existing_points:
            return p

def _local_search(points, m, max_iters, rng, time_limit):
    """Local search: try replacing one point at a time to improve score."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    best_score = _score_configuration(points, m)
    best_points = points.copy()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    # Determine coordinate range
    min_coord = int(points.min()) - 5
    max_coord = int(points.max()) + 5
    # Clamp to reasonable range
    min_coord = max(min_coord, -50)
    max_coord = min(max_coord, 50)
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to replace
        idx = rng.randint(0, n - 1)
        old_point = tuple(points[idx])
        existing = set(map(tuple, points))
        
        # Try several candidate replacements
        best_new_score = best_score
        best_new_point = None
        
        for _ in range(20):
            new_point = _random_point_in_range(min_coord, max_coord, existing, rng)
            points[idx] = new_point
            score = _score_configuration(points, m)
            if score > best_new_score:
                best_new_score = score
                best_new_point = new_point
            points[idx] = old_point
        
        if best_new_point is not None:
            points[idx] = best_new_point
            best_score = best_new_score
            best_points = points.copy()
            no_improve_count = 0
            # Update range
            min_coord = max(int(points.min()) - 5, -50)
            max_coord = min(int(points.max()) + 5, 50)
        else:
            no_improve_count += 1
    
    return best_points, best_score

def _initial_grid_config():
    """Generate the baseline grid configuration."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _initial_nonrect_config():
    """Generate a non-rectangular initial configuration."""
    # 7x8 grid minus 2 points = 54
    points_list = []
    for x in range(1, 8):
        for y in range(1, 9):
            points_list.append((x, y))
    # Remove 2 points randomly
    random.seed(42)
    remove_idx = random.sample(range(len(points_list)), 2)
    remove_set = set(remove_idx)
    points_list = [p for i, p in enumerate(points_list) if i not in remove_set]
    return np.array(points_list, dtype=np.int64)

def _initial_mixed_config():
    """Generate a mixed initial configuration with some structure."""
    # Start with 6x9 grid = 54
    points_list = []
    for x in range(1, 7):
        for y in range(1, 10):
            points_list.append((x, y))
    return np.array(points_list, dtype=np.int64)

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 points and 27 lines
    to maximize point-line incidences.
    """
    start_time = time.time()
    time_limit = 580  # Leave 20s safety margin
    
    rng = random.Random(12345)
    
    # Initial configurations
    configs = []
    
    # Grid 3x18
    configs.append(_initial_grid_config())
    
    # Grid 6x9
    configs.append(_initial_mixed_config())
    
    # Non-rectangular 7x8 minus 2
    configs.append(_initial_nonrect_config())
    
    # Another non-rectangular: 5x10 plus 4
    points_list = []
    for x in range(1, 6):
        for y in range(1, 11):
            points_list.append((x, y))
    # Add 4 more points
    extra = [(6, 1), (6, 2), (7, 1), (7, 2)]
    points_list.extend(extra)
    configs.append(np.array(points_list, dtype=np.int64))
    
    # Evaluate initial configs
    best_points = None
    best_score = -1
    for pts in configs:
        score = _score_configuration(pts, 27)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Local search from best config
    if time.time() - start_time < time_limit:
        searched_points, searched_score = _local_search(
            best_points, 27, max_iters=1000, rng=rng,
            time_limit=time_limit - (time.time() - start_time)
        )
        if searched_score > best_score:
            best_score = searched_score
            best_points = searched_points
    
    # Try local search from other configs if time remains
    for pts in configs:
        if time.time() - start_time > time_limit - 60:
            break
        if np.array_equal(pts, best_points):
            continue
        searched_points, searched_score = _local_search(
            pts, 27, max_iters=500, rng=rng,
            time_limit=max(10, time_limit - (time.time() - start_time))
        )
        if searched_score > best_score:
            best_score = searched_score
            best_points = searched_points
    
    # Generate top 27 lines for best configuration
    top = _top_lines(best_points, 27)
    
    # If we have fewer than 27 lines, add more
    if len(top) < 27:
        # Add lines with fewer incidences
        line_to_points = _all_lines_through_pairs(best_points)
        all_lines = sorted(line_to_points.items(), key=lambda x: -len(x[1]))
        for key, pts in all_lines:
            if len(top) >= 27:
                break
            if key not in [t[0] for t in top]:
                top.append((key, len(pts)))
    
    # Take exactly 27 lines
    top = top[:27]
    
    lines = np.array([t[0] for t in top], dtype=np.int64)
    
    # Ensure we have exactly 27 lines (pad if needed)
    if lines.shape[0] < 27:
        # Add arbitrary lines
        extra = []
        for i in range(27 - lines.shape[0]):
            extra.append((1, 0, -(100 + i)))
        lines = np.vstack([lines, np.array(extra, dtype=np.int64)])
    
    # Ensure points are exactly 54
    if best_points.shape[0] != 54:
        # This shouldn't happen, but handle it
        if best_points.shape[0] > 54:
            best_points = best_points[:54]
        else:
            while best_points.shape[0] < 54:
                best_points = np.vstack([best_points, [[100 + best_points.shape[0], 0]]])
    
    # Final check: ensure no duplicate points
    # (should be fine from construction)
    
    # Check time
    if time.time() - start_time > time_limit:
        # Return the grid config as fallback
        points = _initial_grid_config()
        k = 3
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines
    
    return best_points, lines

# EVOLVE-BLOCK-END
